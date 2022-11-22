import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset

import os
import random
from collections import defaultdict
from PIL import Image
from mpl_toolkits.axes_grid1 import ImageGrid
from loguru import logger
from tqdm import tqdm

random.seed(0) #for reproducibility

import sys

sys.path.insert(0, "./")

import data_tools.UniformSampler as US
import data_tools.Market1501_New as MKT
import data_tools.RandomErase as RE
import train_tools.CenterLoss as CL
import train_tools.TripletLoss as TL
import train_tools.BatchHardMining as BHM

import net as N
import train_new as t

ROOT = "../data/Market-1501"
P = 16
K = 4
EMBEDDING_DIM = 2048 #from resnet50
STARTING_LR = 0.00035 #from bag of tricks paper
NUM_EPOCHS = 120
SHOULD_LOG = False






def main():
    
    transform = T.Compose([
            T.ToTensor(),
            T.Resize(size=(256,128)),
            T.Pad(padding=10, fill=0),
            T.RandomCrop(size=(256,128)),
            T.RandomErasing()
    ])
    
    test_transform = T.Compose([T.ToTensor(), T.Resize(size=(256,128))])
    

    






if __name__ == "__main__":
    main()


