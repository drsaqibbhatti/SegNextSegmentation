import torch
import numpy as np
import os
import json
import cv2
from torch.utils.data import Dataset
from PIL import Image
import csv 
from pycocotools import mask as coco_mask

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, annotation_path, transform=None):
        """
        Args:
            image_dir (str): Path to the images directory.
            annotation_path (str): Path to the COCO JSON annotations.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.image_dir = image_dir
        self.transform = transform
        
        # Load COCO annotations
        with open(annotation_path, 'r') as f:
            self.coco_data = json.load(f)
        
        # Create a mapping from COCO category IDs to a sequential range [0, 79]
        self.category_id_map = {cat_id: idx for idx, cat_id in enumerate(range(1, 81))}

        # Create a mapping from image_id to list of annotations for faster access
        self.image_id_to_annos = {}
        for annotation in self.coco_data['annotations']:
            image_id = annotation['image_id']
            if image_id not in self.image_id_to_annos:
                self.image_id_to_annos[image_id] = []
            self.image_id_to_annos[image_id].append(annotation)
        
        # Load images metadata
        self.images = self.coco_data['images']

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load image
        image_info = self.images[idx]
        image_id = image_info['id']
        image_path = os.path.join(self.image_dir, image_info['file_name'])
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)[:, :, ::-1]  # Convert RGB to BGR
        image = Image.fromarray(image)

        original_width, original_height = image.size
        
        # Initialize an empty mask with shape (80, height, width)
        mask = np.zeros((80, original_height, original_width), dtype=np.uint8)
        
        # Fill the mask with category-specific annotations
        if image_id in self.image_id_to_annos:
            for anno in self.image_id_to_annos[image_id]:
                category_id = anno['category_id']
                
                # Only include relevant categories (1 to 80)
                if category_id in self.category_id_map:
                    mapped_category = self.category_id_map[category_id]
                    
                    if isinstance(anno['segmentation'], dict) and 'counts' in anno['segmentation']:
                        # Decode RLE
                        if isinstance(anno['segmentation']['counts'], list):
                            rle = coco_mask.frPyObjects(anno['segmentation'], original_height, original_width)
                            rle_mask = coco_mask.decode(rle).astype(np.uint8)
                        else:
                            rle_mask = coco_mask.decode(anno['segmentation']).astype(np.uint8)
                    else:
                        # Handle polygons
                        rle_mask = np.zeros((original_height, original_width), dtype=np.uint8)
                        for polygon in anno['segmentation']:
                            pts = np.array(polygon, dtype=np.int32).reshape(-1, 2)
                            cv2.fillPoly(rle_mask, [pts], color=1)
                    
                    # Assign the mask for the mapped category
                    mask[mapped_category] = np.maximum(mask[mapped_category], rle_mask)
        
        # Resize and apply transformations
        resize_height, resize_width = self.transform.transforms[0].size
        image = image.resize((resize_width, resize_height), Image.NEAREST)
        mask_resized = np.zeros((80, resize_height, resize_width), dtype=np.uint8)
        for channel in range(mask.shape[0]):
            mask_resized[channel] = cv2.resize(mask[channel], (resize_width, resize_height), interpolation=cv2.INTER_NEAREST)
        
        image = self.transform(image) if self.transform else image
        mask_tensor = torch.tensor(mask_resized, dtype=torch.long)
        
        return image, mask_tensor

# class SegmentationDataset(Dataset):
#     def __init__(self, image_dir, annotation_path, transform=None):
#         """
#         Args:
#             image_dir (str): Path to the images directory.
#             annotation_path (str): Path to the COCO JSON annotations.
#             transform (callable, optional): Optional transform to be applied on an image.
#         """
#         self.image_dir = image_dir
#         self.transform = transform
        
#         # Load COCO annotations
#         with open(annotation_path, 'r') as f:
#             self.coco_data = json.load(f)
        
#         # Create a mapping from image_id to list of annotations for faster access
#         self.image_id_to_annos = {}
#         for annotation in self.coco_data['annotations']:
#             image_id = annotation['image_id']
#             if image_id not in self.image_id_to_annos:
#                 self.image_id_to_annos[image_id] = []
#             self.image_id_to_annos[image_id].append(annotation)
        
#         # Load images metadata
#         self.images = self.coco_data['images']

#     def __len__(self):
#         return len(self.images)
    
#     def __getitem__(self, idx):
#         # Load image
#         image_info = self.images[idx]
#         image_id = image_info['id']
#         image_path = os.path.join(self.image_dir, image_info['file_name'])
#         image = Image.open(image_path).convert('RGB')
        
#         original_width, original_height = image.size
        
#         # Initialize an empty mask with shape (num_categories, height, width)
#         max_category_id = max([anno['category_id'] for anno in self.coco_data['annotations']])
#         mask = np.zeros((max_category_id + 1, original_height, original_width), dtype=np.uint8)
        
#         # # Overlay each category’s annotation onto the mask
#         # if image_id in self.image_id_to_annos:
#         #     for anno in self.image_id_to_annos[image_id]:
#         #         category_id = anno['category_id']
#         #         polygons = anno['segmentation']
                
#         #         for polygon in polygons:
#         #             pts = np.array(polygon, dtype=np.int32).reshape(-1, 2)
#         #             cv2.fillPoly(mask[category_id], [pts], color=1)
#         if image_id in self.image_id_to_annos:
#             for anno in self.image_id_to_annos[image_id]:
#                 category_id = anno['category_id']
                
#                 if isinstance(anno['segmentation'], dict) and 'counts' in anno['segmentation']:
#                     # Check if 'counts' is a list (uncompressed RLE), then convert to compressed format
#                     if isinstance(anno['segmentation']['counts'], list):
#                         # Convert uncompressed RLE to compressed RLE format for decoding
#                         rle = coco_mask.frPyObjects(anno['segmentation'], original_height, original_width)
#                         rle_mask = coco_mask.decode(rle).astype(np.uint8)
#                     else:
#                         # Decode directly if already in compressed format
#                         rle_mask = coco_mask.decode(anno['segmentation']).astype(np.uint8)
#                 else:
#                     # Handle polygons as before
#                     rle_mask = np.zeros((original_height, original_width), dtype=np.uint8)
#                     for polygon in anno['segmentation']:
#                         pts = np.array(polygon, dtype=np.int32).reshape(-1, 2)
#                         cv2.fillPoly(rle_mask, [pts], color=1)
                
#                 # Add the mask for the specific category
#                 mask[category_id] = np.maximum(mask[category_id], rle_mask)
        
#         # Resize and apply transformations
#         resize_height, resize_width = self.transform.transforms[0].size
#         image = image.resize((resize_width, resize_height), Image.NEAREST)
#         mask_resized = np.zeros((max_category_id + 1, resize_height, resize_width), dtype=np.uint8)
#         for channel in range(mask.shape[0]):
#             mask_resized[channel] = cv2.resize(mask[channel], (resize_width, resize_height), interpolation=cv2.INTER_NEAREST)
        
#         image = self.transform(image) if self.transform else image
#         mask_tensor = torch.tensor(mask_resized, dtype=torch.long)

#         print(f"Image shape after transform: {image.shape}")  # Should be [C, H, W]
#         print(f"Mask tensor shape after resizing: {mask_tensor.shape}") 
#         return image, mask_tensor
    
def generate_csv(self, csv_path="segmentation_data.csv"):
    """
    Generates a CSV file listing each image_id and the unique category_ids present in its mask.
    
    Args:
        csv_path (str): The file path for saving the generated CSV.
    """
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['image_id', 'category_ids'])  # Write header
        
        for image_info in self.images:
            image_id = image_info['id']
            category_ids = set()
            
            if image_id in self.image_id_to_annos:
                for anno in self.image_id_to_annos[image_id]:
                    category_ids.add(anno['category_id'])  # Collect unique category IDs
            
            writer.writerow([image_id, list(category_ids)])  # Write image_id and unique category_ids as a list

    print(f"CSV file saved as {csv_path}")