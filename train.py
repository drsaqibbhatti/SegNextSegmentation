import torch
import random
import numpy as np
import time
import datetime
from datetime import date
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from model.SegNext import SegNext_v1
from torchsummary import summary
from utils.helper import IOU, DiceLoss, FocalTverskyLoss
from dataloader.SegmentationDataset import SegmentationDataset, generate_csv
import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from collections import OrderedDict

def train():
    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda" if USE_CUDA else "cpu")
    print("The device is:", device)

    # Seed for reproducibility
    random.seed(777)
    torch.manual_seed(777)
    if device == 'cuda':
        torch.cuda.manual_seed_all(777)

    # Hyperparameters
    imageWidth = 512
    imageHeight = 512
    batchSize = 32
    learningRate = 0.003
    epochs = 2
    targetAccuracy = 0.991
    num_classes = 80  # COCO segmentation classes

    transformAugCollection = [
        transforms.Resize((imageHeight, imageWidth), interpolation=Image.NEAREST),
        transforms.ToTensor()
    ]
    transAugProcess = transforms.Compose(transformAugCollection)

    transformNormalCollection = [
        transforms.Resize((imageHeight, imageWidth), interpolation=Image.NEAREST),
        transforms.ToTensor()
    ]
    transNormalProcess = transforms.Compose(transformNormalCollection)

    trainDataset = SegmentationDataset(
        image_dir="/home2/Read_only_Folder/COCO_Dataset_bbox_plus_segmenration annotated/train2017/train2017",
        annotation_path="/home2/Read_only_Folder/COCO_Dataset_bbox_plus_segmenration annotated/annotations_trainval2017/annotations/instances_train2017.json",
        transform=transAugProcess
    )
    trainDatasetLoader = DataLoader(trainDataset, batch_size=batchSize, shuffle=True, drop_last=True)

    validDataset = SegmentationDataset(
        image_dir="/home2/Read_only_Folder/COCO_Dataset_bbox_plus_segmenration annotated/val2017/val2017",
        annotation_path="/home2/Read_only_Folder/COCO_Dataset_bbox_plus_segmenration annotated/annotations_trainval2017/annotations/instances_val2017.json",
        transform=transNormalProcess
    )
    validDatasetLoader = DataLoader(validDataset, batch_size=1, shuffle=True, drop_last=False)

    CustomSegmentation = SegNext_v1(
        inputWidth=imageWidth,
        inputHeight=imageHeight,
        class_num=num_classes
        ).to(device)

    print('==== model info ====')
    summary(CustomSegmentation, (3, imageHeight, imageWidth))
    print('====================')

    # Loss function and optimizer
    #loss_fn = FocalTverskyLoss(alpha=0.7, beta=0.3, gamma=2).to(device)
    loss_dice_fn = DiceLoss()

    CustomSegmentation.train()
    optimizer = torch.optim.RAdam(CustomSegmentation.parameters(), lr=learningRate)

    # Directory to save model
    Base_dir = '/home/saqib/deeplearningresearch/python/project/Pre_Training/Trained_Models/Segmentation'
    model_name = "SegNext_v1"
    build_date = str(date.today())
    model_dir = os.path.join(Base_dir, model_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    generate_csv(trainDataset, os.path.join(model_dir, 'train_dataset.csv'))
    generate_csv(validDataset, os.path.join(model_dir, 'valid_dataset.csv'))

    best_model_path = None
    best_train_acc = 0

    existing_runs = [d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))]
    run_number = len(existing_runs) + 1
    run_dir = os.path.join(model_dir, f"run_{run_number}")
    os.makedirs(run_dir)


    metrics = []
    Total_training_time = 0
    for epoch in range(epochs):
        start = datetime.datetime.now()
        batch_costs = []
        batch_accuracies = []
        with tqdm(total=len(trainDatasetLoader), desc=f"Epoch {epoch+1}/{epochs}", unit="batch") as pbar:
            for X, Y in trainDatasetLoader:
                gpu_X = X.to(device)
                gpu_Y = Y.to(device)
                
                # Training step
                CustomSegmentation.train()
                optimizer.zero_grad()
                hypothesis = CustomSegmentation(gpu_X)
                cost = loss_dice_fn(hypothesis, gpu_Y)
                cost.backward()
                optimizer.step()

                # Calculate metrics
                batch_costs.append(cost.cpu().detach().numpy())
                accuracy = IOU(gpu_Y.cpu().detach().numpy(), hypothesis.cpu().detach().numpy())
                batch_accuracies.append(accuracy)
                pbar.update(1)
                pbar.set_postfix(cost=cost.item(), accuracy=accuracy)

        # Epoch metrics
        avg_cost = np.mean(batch_costs)
        avg_acc = np.mean(batch_accuracies)
        end = datetime.datetime.now()
        time_per_epoch = end - start
        Total_training_time += time_per_epoch.total_seconds()

        # Validation step
        val_costs = []
        val_accuracies = []
        CustomSegmentation.eval()
        with torch.no_grad():
            for X_val, Y_val in validDatasetLoader:
                gpu_X_val = X_val.to(device)
                gpu_Y_val = Y_val.to(device)
                
                val_output = CustomSegmentation(gpu_X_val)
                val_loss = loss_dice_fn(val_output, gpu_Y_val)
                val_accuracy = IOU(gpu_Y_val.cpu().detach().numpy(), val_output.cpu().detach().numpy())
                
                val_costs.append(val_loss.cpu().detach().numpy())
                val_accuracies.append(val_accuracy)

        val_cost = np.mean(val_costs)
        val_acc = np.mean(val_accuracies)

        print(f'Epoch: {epoch + 1}, Train cost: {avg_cost:.5f}, Train accuracy: {avg_acc:.5f}, Val cost: {val_cost:.5f}, Val accuracy: {val_acc:.5f}')




        if avg_acc > best_train_acc:
            if best_model_path is not None:
                os.remove(best_model_path)
            best_train_acc = avg_acc
            best_model_path = os.path.join(run_dir,f"{model_name}_train_{best_train_acc:.5f}_loss_{avg_cost:.5f}_epoch_{epoch+1}_lr_{learningRate:.3f}_time_{Total_training_time / 60:.2f}M_alpha_{alphaa}_beta_{betaa}_gamma_{gammaa}_best.pth")
            torch.save(CustomSegmentation, best_model_path)
            print(f"Saved best model at epoch {epoch+1} with training accuracy: {best_train_acc:.9f}")
            
        # Early stopping based on validation accuracy
        if avg_acc > targetAccuracy:  # Use train accuracy for early stopping
            final_accuracy = avg_acc
            final_cost = avg_cost
            print(f'Target accuracy achieved at epoch number {epoch + 1}', ', final train accuracy=', final_accuracy, ', final cost=', final_cost)
            break
        # Append metrics to the list for CSV
        metrics.append({
            'Epoch': epoch + 1,
            'Train Accuracy': avg_acc,
            'Validation Accuracy': val_acc,
            'Train Cost': avg_cost,
            'Validation Cost': val_cost,
            'Learning Rate': learningRate
        })
        
        if epoch%5==0:
            metrics_df = pd.DataFrame(metrics)
            csv_path = os.path.join(run_dir, "training_metrics.csv")
            metrics_df.to_csv(csv_path, index=False)
            print(f"Saved training metrics till epoch:{epoch} to {csv_path}")
            plt.figure(figsize=(10, 6))
            plt.plot(metrics_df['Epoch'], metrics_df['Train Accuracy'], label='Training Accuracy')
            plt.plot(metrics_df['Epoch'], metrics_df['Validation Accuracy'], label='Validation Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.title('Training and Validation Accuracy')
            # plt.ylim(0.001, 0.99)
            plt.legend()
            plt.grid(True)
            # Save the figure without displaying it
            accuracy_fig_path = os.path.join(run_dir, 'accuracy_plot.png')
            plt.savefig(accuracy_fig_path)
            plt.close()  # Close the figure to free up memory
            print(f'Saved accuracy plot to {accuracy_fig_path}')
            
            # Plot Training and Validation Loss
            plt.figure(figsize=(10, 6))
            plt.plot(metrics_df['Epoch'], metrics_df['Train Cost'], label='Training Loss')
            plt.plot(metrics_df['Epoch'], metrics_df['Validation Cost'], label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss')
            plt.legend()
            plt.grid(True)

            # Save the figure in the same directory
            loss_fig_path = os.path.join(run_dir, 'loss_plot.png')
            plt.savefig(loss_fig_path)
            plt.close()  # Close the figure to free up memory
            print(f'Saved loss plot to {loss_fig_path}')

            
    print(f"Total training time: {Total_training_time / 60:.2f} minutes")
    # model save
    # CustomSegmentation.eval()
    # compiled_model = torch.jit.script(CustomSegmentation)
    # torch.jit.save(compiled_model, "E://busbar_al_welding_prediction.pt")


    last_model_path = os.path.join(run_dir,f"{model_name}_train_{avg_acc:.5f}_loss_{avg_cost:.5f}_epoch_{epoch+1}_lr_{learningRate:.3f}_time_{Total_training_time / 60:.2f}M_alpha_{alphaa}_beta_{betaa}_gamma_{gammaa}_last.pth")
    torch.save(CustomSegmentation, last_model_path)
    print(f"Saved last model after epoch {epoch+1}")
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame(metrics)
    csv_path = os.path.join(run_dir, "training_metrics.csv")
    metrics_df.to_csv(csv_path, index=False)
    print(f"Saved training metrics to {csv_path}")
    
        # Plot Training and Validation Accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(metrics_df['Epoch'], metrics_df['Train Accuracy'], label='Training Accuracy')
    plt.plot(metrics_df['Epoch'], metrics_df['Validation Accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.ylim(0.4, 0.99)
    plt.legend()
    plt.grid(True)

    # Save the figure in the same directory
    accuracy_fig_path = os.path.join(run_dir, 'accuracy_plot.png')
    plt.savefig(accuracy_fig_path)
    print(f'Saved accuracy plot to {accuracy_fig_path}')

    # Plot Training and Validation Loss
    plt.figure(figsize=(10, 6))
    plt.plot(metrics_df['Epoch'], metrics_df['Train Cost'], label='Training Loss')
    plt.plot(metrics_df['Epoch'], metrics_df['Validation Cost'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    # Save the figure in the same directory
    loss_fig_path = os.path.join(run_dir, 'loss_plot.png')
    plt.savefig(loss_fig_path)
    print(f'Saved loss plot to {loss_fig_path}')

    # Plot Learning Rate
    plt.figure(figsize=(10, 6))
    plt.plot(metrics_df['Epoch'], metrics_df['Learning Rate'], label='Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.grid(True)

    # Save the figure in the same directory
    lr_fig_path = os.path.join(run_dir, 'learning_rate_plot.png')
    plt.savefig(lr_fig_path)
    print(f'Saved learning rate plot to {lr_fig_path}')

    plt.show()
if __name__ == '__main__':
    train()