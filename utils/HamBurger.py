import math
import torch
from functools import partial
from torch import nn
import torch.nn.functional as F




class _MatrixDecomposition2DBase(nn.Module):
    '''
    Base class for furhter implementing the NMF, VQ or CD as in paper
    
    https://arxiv.org/pdf/2109.04553.pdf

    this script only has NMF as it has best performance for semantic segmentation
    as mentioned in paper

    D (dictionery) in paper is bases 
    C (codes) in paper is coef here
    '''
    def __init__(self, 
                 in_channels=512,
                 rank=64,
                 steps=6):
        super().__init__()

        self.spatial = True

        self.S = 1
        self.D = in_channels
        self.R = rank

        self.train_steps = steps
        self.eval_steps = steps

        self.inv_t = 1
        self.eta = 0.9

        self.rand_init = True
        print(30*'=')
        print('spatial: ', self.spatial)
        print('S: ', self.S)
        print('D: ', self.D)
        print('R: ', self.R)
        print('train_steps: ', self.train_steps)
        print('eval_steps: ', self.eval_steps)
        print('inv_t: ', self.inv_t)
        print('eta: ', self.eta)
        print('rand_init: ', self.rand_init)
        print(30*'=')

    def _bild_bases(self, B,S,D,R):
        raise NotImplementedError

    def local_setp(self, x, bases, coef):
        raise NotImplementedError

    def compute_coef(self, x, bases, coef):
        raise NotImplementedError

    @torch.no_grad()
    def local_inference(self, x, bases):
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        # here: N = HW and D = C in case of spatial attention
        coef = torch.bmm(x.transpose(1,2), bases)
        # column wise softmax ignore batch dim, i.e, on HW dim
        coef = F.softmax(self.inv_t*coef, dim=-1)

        steps = self.train_steps if self.training else self.eval_steps
        for _ in range(steps):
            bases, coef = self.local_setp(x, bases, coef)
        return bases, coef
    
    @torch.no_grad()
    def online_update(self, bases):
        update = bases.mean(dim=0)
        self.bases += self.eta * (update - self.bases)
        # column wise normalization i.e. HW dim
        self.bases = F.normalize(self.bases, dim=1)
        return None


    def forward(self, x, return_bases=False):
        B, C, H, W = x.shape

        if self.spatial:
            # spatial attention k
            D = C // self.S # reduce channels
            N = H * W
            x = x.view(B * self.S, D, N)
        else:
            D = H * W
            N = C // self.S
            x = x.view(B * self.S, N, D).transpose(1, 2)
        
        if not self.rand_init and not hasattr(self, 'bases'):
            bases = self._build_bases(1, self.S, D, self.R)
            self.register_buffer('bases', bases)
        # (S, D, R) -> (B * S, D, R)
        if self.rand_init:
            bases = self._build_bases(B, self.S, D, self.R)
        else:
            bases = self.bases.repeat(B,1,1)

        bases, coef = self.local_inference(x, bases)
        # (B * S, N, R)
        coef = self.compute_coef(x, bases, coef)

         # (B * S, D, R) @ (B * S, N, R)^T -> (B * S, D, N)
        x = torch.bmm(bases, coef.transpose(1,2))

        # (B * S, D, N) -> (B, C, H, W)
        if self.spatial:
            x = x.view(B, C, H, W)
        else:
            x = x.transpose(1,2).view(B, C, H, W)

        bases = bases.view(B, self.S, D, self.R)

        if not self.rand_init and not self.training and not return_bases:
            self.online_update(bases)

        return x

class NMF2D(_MatrixDecomposition2DBase):
    def __init__(self, 
                 in_channels=512,
                 rank=64,
                 steps=6):
        super().__init__(in_channels=in_channels,
                         rank=rank,
                         steps=steps)

        self.inv_t = 1

    def _build_bases(self, B, S, D, R):
        
        bases = torch.rand((B*S, D, R)).to('cuda' if torch.cuda.is_available() else 'cpu')
        bases = F.normalize(bases, dim=1) # column wise normalization i.e HW dim

        return bases
    
    @torch.no_grad()
    def local_setp(self, x, bases, coef):
        '''
        Algorithm 2 in paper
        NMF with multiliplicative update.
        '''
        # coef (C/codes)update
        # (B*S, D, N)T @ (B*S, D, R) -> (B*S, N, R)
        numerator = torch.bmm(x.transpose(1,2), bases) # D^T @ X
        # (BS, N, R) @ [(BS, D, R)T @ (BS, D, R)] -> (BS, N, R)
        denominator = coef.bmm(bases.transpose(1,2).bmm(bases)) # D^T @ D @ C
        # Multiplicative update
        coef = coef * (numerator / (denominator + 1e-7)) # updated C    
        # bases (D/dict) update
        # (BS, D, N) @ (BS, N, R) -> (BS, D, R)
        numerator = torch.bmm(x, coef) # X @ C^T
        # (BS, D, R) @ [(BS, D, R)T @ (BS, D, R)] -> (BS, D, R)
        denominator = bases.bmm(coef.transpose(1,2).bmm(coef)) # D @ D @ C^T
        # Multiplicative update
        bases = bases * (numerator / (denominator + 1e-7)) # updated D
        return bases, coef

    def compute_coef(self, x, bases, coef):
        # (B*S, D, N)T @ (B*S, D, R) -> (B*S, N, R)
        numerator = torch.bmm(x.transpose(1,2), bases) # D^T @ X
        # (BS, N, R) @ [(BS, D, R)T @ (BS, D, R)] -> (BS, N, R)
        denominator = coef.bmm(bases.transpose(1,2).bmm(bases)) # D^T @ D @ C
        # Multiplicative update
        coef = coef * (numerator / (denominator + 1e-7))
        return coef

'''
Make Burger
'''
class HamBurger(nn.Module):
    def __init__(self, 
                 in_channels=512,
                 steps=6,
                 rank=64):
        super().__init__()
        self.put_cheese = True

        # add Relu at end as NMF works of non-negative only
        self.lower_bread = torch.nn.Sequential(torch.nn.Conv2d(in_channels, in_channels, 1),
                                               torch.nn.ReLU(inplace=True))
        self.ham = NMF2D(in_channels=in_channels,
                         steps=steps,
                         rank=rank)
        self.cheese = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels, 
                            out_channels=in_channels,
                            kernel_size=3,
                            padding=1),
            torch.nn.BatchNorm2d(in_channels),
            torch.nn.ReLU(),
        )
        self.upper_bread = nn.Conv2d(in_channels, in_channels, 1, bias=False)


    def forward(self, x):
        skip = x.clone()

        x = self.lower_bread(x)
        x = self.ham(x)

        if self.put_cheese:
            x = self.cheese(x)
        
        x = self.upper_bread(x)
        x = F.relu(x + skip, inplace=True)

        return x

    def online_update(self, bases):
        if hasattr(self.ham, 'online_update'):
            self.ham.online_update(bases)