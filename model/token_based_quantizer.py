import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as distributed
from typing import List, Optional
import numpy as np


def l2norm(t):
    return F.normalize(t, p = 2, dim = -1)


class TokenBasedMSVectorQuantizer(nn.Module):
    def __init__(self, 
                 n_embed,
                 embedding_dim,
                 beta,
                 decay=0.99, 
                 eps=1e-5, 
                 statistic_code_usage=True, 
                 kmeans_init=False, 
                 codebook_init_path='',
                 tokens_per_window=16):
        super().__init__()
        self.num_tokens = n_embed
        self.codebook_dim = embedding_dim
        self.beta = beta
        self.decay = decay
        self.tokens_per_window = tokens_per_window
        
        self.embedding = nn.Embedding(self.num_tokens, self.codebook_dim)
        self.init_vocab(-1)
        
        self.statistic_code_usage = statistic_code_usage
        if statistic_code_usage:
            self.register_buffer('cluster_size', torch.zeros(n_embed))
        
        if distributed.is_available() and distributed.is_initialized():
            print("ddp is enable, so use ddp_reduce to sync the statistic_code_usage for each gpu!")
            self.all_reduce_fn = distributed.all_reduce
        else:
            self.all_reduce_fn = nn.Identity()
    
    def init_vocab(self, eini: float):
        if eini > 0:
            nn.init.trunc_normal_(self.embedding.weight.data, std=eini)
        elif eini < 0:
            base = self.codebook_dim ** -0.5
            base /= 36
            self.embedding.weight.data.uniform_(-abs(eini) * base, abs(eini) * base)
    
    def eini(self, eini):
        if eini > 0: 
            nn.init.trunc_normal_(self.embedding.weight.data, std=eini)
        elif eini < 0: 
            self.embedding.weight.data.uniform_(-abs(eini) / self.num_tokens, abs(eini) / self.num_tokens)
    
    def extra_repr(self) -> str:
        return f'tokens_per_window={self.tokens_per_window}, beta={self.beta}'
    
    def print_codebook_usage(self, sync_ddp=True):
        if not self.statistic_code_usage:
            return
        
        cluster_size_tensor = self.cluster_size.clone()
        if sync_ddp and distributed.is_available() and distributed.is_initialized():
            distributed.all_reduce(cluster_size_tensor, op=distributed.ReduceOp.SUM)
            cluster_size_tensor /= distributed.get_world_size()
        
        cluster_size = cluster_size_tensor.cpu().numpy()
        active = (cluster_size > 0).sum()
        total = len(cluster_size)
        util = active / total * 100
        
        print(f"📊 Codebook: {active}/{total} ({util:.1f}%) | "
              f"Mean: {cluster_size.mean():.2f} | "
              f"Max: {cluster_size.max():.2f} | "
              f"Dead: {total - active}")
    
    def forward(self, z):
        B, N_total, D = z.shape
        
        z = l2norm(z)
        
        L = (B * N_total) // self.tokens_per_window
        z_LND = z.view(L, self.tokens_per_window, D)
        
        z_LND_no_grad = z_LND.detach()
        
        with torch.amp.autocast('cuda', enabled=False):
            z_flat = z_LND.reshape(-1, D)
            z_flat_no_grad = z_LND_no_grad.reshape(-1, D)
            
            d = torch.sum(z_flat_no_grad.square(), dim=1, keepdim=True) + \
                torch.sum(self.embedding.weight.data.square(), dim=1, keepdim=False)
            d.addmm_(z_flat_no_grad, self.embedding.weight.data.T, alpha=-2, beta=1)

            idx_flat = torch.argmin(d, dim=1)
            idx_L16 = idx_flat.view(L, self.tokens_per_window)
            
            if self.statistic_code_usage:
                idx_onehot = F.one_hot(idx_flat, num_classes=self.num_tokens).float()
                cluster_size = idx_onehot.sum(0)
                self.all_reduce_fn(cluster_size)
                self.cluster_size.data.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)
            
            z_q_flat = self.embedding(idx_flat)
            z_q_LND = z_q_flat.view(L, self.tokens_per_window, D)
            
            commitment_loss = F.mse_loss(z_LND, z_q_LND.detach())
            reconstruction_loss = F.mse_loss(z_LND_no_grad, z_q_LND)
            vq_loss = self.beta * commitment_loss + reconstruction_loss
            
            z_q_LND = z_LND + (z_q_LND - z_LND).detach()
        
        z_q = z_q_LND.reshape(B, N_total, D)
        
        return z_q, vq_loss, idx_L16

