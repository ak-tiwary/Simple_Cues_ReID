

import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter

import os
import random
from collections import defaultdict
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from mpl_toolkits.axes_grid1 import ImageGrid
from loguru import logger
random.seed(0) #for reproducibility

import sys

sys.path.insert(0, "./")

import data_tools.UniformSampler as US
import data_tools.Market1501 as MKT
import data_tools.RandomErase as RE
import train_tools.CenterLoss as CL
import train_tools.TripletLoss as TL
import net as N
import train as t

ROOT = "../data/Market-1501"
P = 16
K = 4
EMBEDDING_DIM = 2048 #from resnet50
STARTING_LR = 0.00035 #from bag of tricks paper
NUM_EPOCHS = 120
SHOULD_LOG = False


import train_tools.BatchHardMining as BHM
from tqdm import tqdm






def main():   
    
      transform = T.Compose([
            T.ToTensor(),
            T.Resize(size=(256,128)),
            T.Pad(padding=10, fill=0),
            T.RandomCrop(size=(256,128)),
            T.RandomErasing()
      ])
    
      # test_transform = T.Compose([
      #       T.ToTensor(),
      #       T.Resize(size=(256,128)),
      #       T.Pad(padding=10, fill=0),
      #       T.RandomCrop(size=(256,128)),
      #       T.RandomErasing()
      # ])
      
      test_transform = T.Compose([T.ToTensor(), T.Resize(size=(256,128))])
      # Get train and test dataset: call market1501() with train=True/False
      train_set = MKT.Market1501(root=ROOT, train=True, transform=transform)
      test_set = MKT.Market1501(root=ROOT, train=False, transform=test_transform)
      
      logger.info(f"len(train_set): {len(train_set)}")
      
      # Form start indices, num classes, num examples from the datasets
      train_labels = train_set.labels
      train_start_indices = {i : train_set.label_to_start_stop_idxs[label][0] 
                        for i,label in enumerate(train_labels)}
      train_num_examples = {i : 
                        (train_set.label_to_start_stop_idxs[label][1] - \
                        train_set.label_to_start_stop_idxs[label][0] + 1)
                        for i, label in enumerate(train_labels)}
      train_num_classes = len(train_labels)
      
            # Form start indices, num classes, num examples from the datasets
      test_labels = test_set.labels
      test_start_indices = {i : test_set.label_to_start_stop_idxs[label][0] 
                        for i,label in enumerate(test_labels)}
      test_num_examples = {i : 
                        (test_set.label_to_start_stop_idxs[label][1] - \
                        test_set.label_to_start_stop_idxs[label][0] + 1)
                        for i, label in enumerate(test_labels)}
      test_num_classes = len(test_labels)
      
      
      

      
      # get the uniform samplers; pass in the dataset
      train_k_at = US._get_k_at_time(start_indices=train_start_indices, num_examples=train_num_examples,
                                    K=K, num_classes=train_num_classes)
      test_k_at = US._get_k_at_time(start_indices=test_start_indices, num_examples=test_num_examples,
                                    K=K, num_classes=test_num_classes)
      train_sampler = US.ClassUniformBatchSampler(
            dataset=train_set, P=P, K=K, k_at_time=train_k_at, num_classes=train_num_classes,
            start_indices=train_start_indices, num_examples=train_num_examples
      )
      
      test_sampler = US.ClassUniformBatchSampler(
            dataset=test_set, P=P, K=K, k_at_time=test_k_at, num_classes=test_num_classes,
            start_indices=test_start_indices, num_examples=test_num_examples
      )
      
      # create dataloaders
      train_loader = DataLoader(dataset=train_set, batch_sampler=train_sampler, num_workers=0, pin_memory=False)
      test_loader = DataLoader(dataset=test_set, batch_sampler=test_sampler, num_workers=0, pin_memory=False)
      
      # Create model
      model = N.Net(num_classes=train_num_classes)
      device = torch.device("cuda")
      model.to(device)
      
      model.load_state_dict(torch.load(""))
      
      
      # Create loss functions, optimizer, scheduler, add optimizer to scheduler
      #triplet_loss = TL.TripletLoss().to(device)
      triplet_loss = nn.SoftMarginLoss()
      center_loss = CL.CenterLoss(num_classes=train_num_classes, embedding_dim=EMBEDDING_DIM).to(device)
      cross_entropy_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
      
      optimizer = torch.optim.Adam(model.parameters(), lr=STARTING_LR, weight_decay=5e-4)
      scheduler1 = lr_scheduler.LambdaLR(optimizer, lr_lambda= lambda epoch: epoch / 10)
      scheduler2_prime = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
      scheduler3_prime = lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=5) #depends on num epochs
      scheduler2 = lr_scheduler.ChainedScheduler([scheduler2_prime, scheduler3_prime])
      scheduler3 = lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=5)
      scheduler = lr_scheduler.SequentialLR(optimizer,
                                                [scheduler1, scheduler2, scheduler3],
                                                milestones=[10, 71])

      # Create summary writer object
      train_writer = SummaryWriter('runs/metric_learning_3')
      test_writer = SummaryWriter('runs/metric_learning_4')
      
      if SHOULD_LOG:
            logger.info(f"Starting training")
      # Call train.py
      t.train(train_loader=train_loader, val_loader=test_loader,model=model, triplet_loss=triplet_loss,
              center_loss=center_loss, cross_entropy_loss=cross_entropy_loss, optimizer=optimizer, 
              scheduler=scheduler, train_summary_writer=train_writer, val_summary_writer=test_writer, device=device, num_epochs=NUM_EPOCHS, should_log=SHOULD_LOG)
      
      # Call summary writer .close()
      train_writer.close()
      test_writer.close()
    
if __name__ == "__main__":
    main()