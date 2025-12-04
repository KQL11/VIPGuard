'''
@File    :   FaceChecker.py
@Author  :   Kaiqing.Lin
@Update  :   2025/04/21
'''

import os
import os.path as osp
import argparse
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import scipy.io as sio
import imageio.v2 as imageio
import cv2
from termcolor import cprint
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
from termcolor import cprint


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, out_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            # nn.LayerNorm(dim, eps=1e-2),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )
        # print(1)

    def init_weight(self):
        for module in self.modules():
            if isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        squeeze_flag = False
        if x.ndim == 2:
            x = x.unsqueeze(dim=0)
            squeeze_flag = True
        for _, layer in enumerate(self.net):
            x = layer(x)
        # out = self.net(x)
        if squeeze_flag:
            x = x.squeeze(dim=0)
        return x


class CrossAttention(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Q, K, V projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        # Shapes
        # query: (batch_size, q_len, embed_dim)
        # key/value: (batch_size, kv_len, embed_dim)
        dim_flag = False
        if query.ndim == 2:
            query = query.unsqueeze(dim=0)
            dim_flag = True
        if key.ndim == 2:
            key = key.unsqueeze(dim=0)
            dim_flag = True
        if value.ndim == 2:
            value = value.unsqueeze(dim=0)
            dim_flag = True
        B, Q_len, _ = query.size()
        _, KV_len, _ = key.size()

        # 1. Linear projections
        Q = self.q_proj(query)  # (B, Q_len, embed_dim)
        K = self.k_proj(key)  # (B, KV_len, embed_dim)
        V = self.v_proj(value)  # (B, KV_len, embed_dim)

        # 2. Split heads
        Q = Q.view(B, Q_len, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, Q_len, D)
        K = K.view(B, KV_len, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, KV_len, D)
        V = V.view(B, KV_len, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, KV_len, D)

        # 3. Scaled Dot-Product Attention
        attn_weights = torch.matmul(Q, K.transpose(-2, -1))  # (B, H, Q_len, KV_len)
        attn_weights = attn_weights / (self.head_dim ** 0.5)

        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))

        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # 4. Attention Output
        attn_output = torch.matmul(attn_probs, V)  # (B, H, Q_len, D)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, Q_len, self.embed_dim)  # (B, Q_len, E)

        # 5. Output projection
        output = self.out_proj(attn_output)  # (B, Q_len, E)

        # Only output the feature (not the attn weight)
        if dim_flag:
            output = output.squeeze(0)
        return output


class FaceChecker(nn.Module):
    def __init__(self, in_dim, depth, num_heads=8, out_dim=3584):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(depth):
            if i != depth - 1:
                self.blocks.append(CrossAttention(in_dim, num_heads, dropout=0.1))
                self.blocks.append(FeedForward(dim=in_dim, hidden_dim=2 * in_dim, out_dim=in_dim, dropout=0.1))
            else:
                self.blocks.append(CrossAttention(in_dim, num_heads, dropout=0.1))
                self.blocks.append(FeedForward(dim=in_dim, hidden_dim=2 * in_dim, out_dim=out_dim, dropout=0.1))

        self.vip_prompt = torch.randn(506, in_dim)

    def forward(self, img_feature_q, img_feature_v):
        feature = img_feature_v
        for block in self.blocks:
            if isinstance(block, CrossAttention):
                feature = block(query=img_feature_q, key=feature, value=feature)
            elif isinstance(block, FeedForward):
                feature = block(feature)
            else:
                AssertionError("Unknown block type.")

        return feature


if __name__ == '__main__':
    pass