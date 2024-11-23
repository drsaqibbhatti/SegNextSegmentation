import torch
import torch.nn as nn

from utils.HamBurger import HamBurger

import torch.nn.functional as F

class LayerScale(torch.nn.Module):

    def __init__(self, inChannels, init_value=1e-2):
        super().__init__()
        self.inChannels = inChannels
        self.init_value = init_value
        self.layer_scale = torch.nn.Parameter(init_value * torch.ones((inChannels)), requires_grad=True)

    def forward(self, x):
        if self.init_value == 0.0:
            return x
        else:
            scale = self.layer_scale.unsqueeze(-1).unsqueeze(-1)
            return scale * x
        
class FFN(torch.nn.Module):

    def __init__(self, in_channels, out_channels, hid_channels):
        super().__init__()
        self.fc1 = torch.nn.Conv2d(in_channels, hid_channels, 1)
        self.dwconv = torch.nn.Conv2d(in_channels=hid_channels,
                                      out_channels=hid_channels,
                                      kernel_size=3,
                                      groups=hid_channels,
                                      padding=1)
        self.act = torch.nn.GELU()
        self.fc2 = torch.nn.Conv2d(hid_channels, out_channels, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class BlockFFN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hid_channels, ls_init_val=1e-2):
        super().__init__()
        self.norm = torch.nn.BatchNorm2d(num_features=in_channels)
        self.ffn = FFN(in_channels, out_channels, hid_channels)
        self.layer_scale = LayerScale(in_channels, init_value=ls_init_val)

    
    def forward(self, x):
        skip = x.clone()

        x = self.norm(x)
        x = self.ffn(x)
        x = self.layer_scale(x)

        op = skip + x
        return op
    
class MSCA(torch.nn.Module):
    def __init__(self, dim):
        super(MSCA, self).__init__()
        # input
        self.conv55 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim) 
        # split into multipats of multiscale attention
        self.conv17_0 = nn.Conv2d(dim, dim, (1,7), padding=(0, 3), groups=dim)
        self.conv17_1 = nn.Conv2d(dim, dim, (7,1), padding=(3, 0), groups=dim)

        self.conv111_0 = nn.Conv2d(dim, dim, (1,11), padding=(0,5), groups=dim)
        self.conv111_1 = nn.Conv2d(dim, dim, (11,1), padding=(5,0), groups=dim)

        self.conv211_0 = nn.Conv2d(dim, dim, (1,21), padding=(0, 10), groups=dim)
        self.conv211_1 = nn.Conv2d(dim, dim, (21,1), padding=(10, 0), groups=dim)

        self.conv11 = nn.Conv2d(dim, dim, 1) # channel mixer

    def forward(self, x):
        
        skip = x.clone()

        c55 = self.conv55(x)
        c17 = self.conv17_0(x)
        c17 = self.conv17_1(c17)
        c111 = self.conv111_0(x)
        c111 = self.conv111_1(c111)
        c211 = self.conv211_0(x)
        c211 = self.conv211_1(c211)

        add = c55 + c17 + c111 + c211

        mixer = self.conv11(add)

        op = mixer * skip

        return op
    

class BlockMSCA(torch.nn.Module):
    def __init__(self, dim, ls_init_val=1e-2):
        super().__init__()
        self.norm = torch.nn.BatchNorm2d(dim)
        self.proj1 = torch.nn.Conv2d(dim, dim, 1)
        self.act = torch.nn.GELU()
        self.msca = MSCA(dim)
        self.proj2 = torch.nn.Conv2d(dim, dim, 1)
        self.layer_scale = LayerScale(dim, init_value=ls_init_val)


    def forward(self, x):
        skip = x.clone()
        x = self.norm(x)
        x = self.proj1(x)
        x = self.act(x)
        x = self.msca(x)
        x = self.proj2(x)
        x = self.layer_scale(x)

        out = x + skip

        return out

class StageMSCA(nn.Module):
    def __init__(self, dim, ffn_ratio=4., ls_init_val=1e-2):
        super().__init__()
        # print(f'StageMSCA {drop_path}')
        self.msca_block = BlockMSCA(dim, ls_init_val)

        ffn_hid_dim = int(dim * ffn_ratio)
        self.ffn_block = BlockFFN(in_channels=dim, out_channels=dim,
                                  hid_channels=ffn_hid_dim, ls_init_val=ls_init_val)

    def forward(self, x): # input coming form Stem
        # B, N, C = x.shape
        # x = x.permute()
        x = self.msca_block(x)
        x = self.ffn_block(x)

        return x
    
class StemConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.feature = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels,
                            out_channels=out_channels //2,
                            kernel_size=3,
                            stride=2,
                            padding=1),
            torch.nn.BatchNorm2d(out_channels //2, eps=1e-5),
            torch.nn.GELU(),
            torch.nn.Conv2d(in_channels=out_channels //2,
                            out_channels=out_channels,
                            kernel_size=3,
                            stride=2,
                            padding=1),
            torch.nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        x = self.feature(x)
        return x
    

    
class DownSample(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.proj = torch.nn.Conv2d(in_channels,
                                    out_channels,
                                    kernel_size=3,
                                    stride=2,
                                    padding=1)

    def forward(self, x):
        x = self.proj(x)
        return x
    
    
class SegNext(torch.nn.Module):

    def __init__(self,
                 inputWidth=512,
                 inputHeight=512,
                 num_class=80,
                 embed_dims=[32, 64, 460, 256],  #32, 64, 128, 128 => 320
                 ffn_ratios=[4, 4, 4, 4],
                 depths=[3, 3, 5, 2],
                 ls_init_val=1e-2):
        super(SegNext, self).__init__()

        self.inputWidth = inputWidth
        self.inputHeight = inputHeight
        self.num_stages = len(embed_dims)
        self.hamburger_input = sum(embed_dims[1:])
        
        #encoding
        #640->320->160->80->40->20
        #    x2,  x4   x8  x16 x32
        for i in range(self.num_stages):

            if i == 0:
                input_embed = StemConv(in_channels=3, out_channels=embed_dims[0])
            else:
                input_embed = DownSample(in_channels=embed_dims[i-1], out_channels=embed_dims[i])

            block = []
            for d in range(depths[i]):
                block.append(StageMSCA(dim=embed_dims[i], ffn_ratio=ffn_ratios[i], ls_init_val=ls_init_val))
            stage = torch.nn.ModuleList(block)
            
            norm_layer = torch.nn.BatchNorm2d(embed_dims[i])


            setattr(self, f'input_embed{i+1}', input_embed)
            setattr(self, f'stage{i+1}', stage)
            setattr(self, f'norm_layer{i+1}', norm_layer)

        #decoding
        self.decoder = HamBurger(in_channels=self.hamburger_input,
                                 steps=6,
                                 rank=64)
        
        self.final = torch.nn.Conv2d(in_channels=self.hamburger_input,
                                     out_channels=num_class,
                                     kernel_size=1)
        
        
        
    def forward(self, x):

        #Features
        features = []

        for i in range(self.num_stages):
            input_embed = getattr(self, f'input_embed{i+1}')
            stage = getattr(self, f'stage{i+1}')
            norm_layer = getattr(self, f'norm_layer{i+1}')
            
            x = input_embed(x)
            
            for stg in stage:
                x = stg(x)
            
            x = norm_layer(x)
            features.append(x)
            
        features = features[1:] # drop stage 1 features b/c low level
        features = [F.interpolate(feature, size=features[-3].shape[2:], mode='bilinear') for feature in features]
        x = torch.cat(features, dim=1)

        x = self.decoder(x)
        x = self.final(x)
        x = F.interpolate(x, size=(self.inputHeight, self.inputWidth), mode='bilinear', align_corners=True)
        return torch.sigmoid(x)



def SegNext_v4(class_num=80,
               inputWidth=640,
               inputHeight=640):
    
    embed_dims=[32, 64, 460, 256]
    ffn_ratios=[4, 4, 4, 4]
    depths=[3, 3, 5, 2]

    return SegNext(inputWidth=inputWidth,
                      inputHeight=inputHeight,
                      num_class=class_num,
                      embed_dims=embed_dims,
                      ffn_ratios=ffn_ratios,
                      depths=depths)


def SegNext_v3(class_num=80,
               inputWidth=640,
               inputHeight=640):
    
    embed_dims=[32, 64, 256, 256]
    ffn_ratios=[3, 3, 3, 3]
    depths=[3, 3, 3, 2]

    return SegNext(inputWidth=inputWidth,
                      inputHeight=inputHeight,
                      num_class=class_num,
                      embed_dims=embed_dims,
                      ffn_ratios=ffn_ratios,
                      depths=depths)


def SegNext_v2(class_num=80,
               inputWidth=640,
               inputHeight=640):
    
    embed_dims=[32, 64, 128, 128]
    ffn_ratios=[2, 2, 2, 2]
    depths=[3, 3, 3, 2]

    return SegNext(inputWidth=inputWidth,
                      inputHeight=inputHeight,
                      num_class=class_num,
                      embed_dims=embed_dims,
                      ffn_ratios=ffn_ratios,
                      depths=depths)

def SegNext_v1(class_num=80,
               inputWidth=640,
               inputHeight=640):
    
    embed_dims=[32, 64, 128, 128]
    ffn_ratios=[2, 2, 2, 2]
    depths=[2, 2, 2, 2]

    return SegNext(inputWidth=inputWidth,
                      inputHeight=inputHeight,
                      num_class=class_num,
                      embed_dims=embed_dims,
                      ffn_ratios=ffn_ratios,
                      depths=depths)