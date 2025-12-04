"""
Author: Kaiqing Lin
Date: 2024/8/8
File: Load_Model.py
"""
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


def load_arcface():
    sys.path.append("/data0/linkaiqing/code/MLLM/VIP_DFD_LLM/Models/Face_Model/ArcFace")
    from iresnet import iresnet100

    arcface = iresnet100(pretrained=False, fp16=False)
    state_dict = torch.load('/data0/linkaiqing/code/MLLM/VIP_DFD_LLM/Models/Face_Model/ArcFace/arcface.pt',
                   map_location='cpu')
    arcface.load_state_dict(state_dict, strict=True)
    print("Load Arcface")
    return arcface.eval()

if __name__ == '__main__':
    pass
