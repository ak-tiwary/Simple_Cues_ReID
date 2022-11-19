import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.tensorboard as tensorborad

import train_tools.BatchHardMining as BHM


def train(train_loader, test_loader, model, triplet_loss, center_loss,
          cross_entropy_loss, optimizer, scheduler, summary_writer,
          device=torch.device("cuda"), num_epochs=1, beta=0.0005):
    """model is the network. scheduler is the learning rate scheduler,
    summary_writer is the tensorboard summary writer.
    
    Beta is a hyperparameter, set to 0.0005 from the Bag of Tricks paper."""
    
    #has a learnable parameter
    center_loss = center_loss.to(device)
    
    for i in range(num_epochs):
          #X is Nx3xHxW augmented. y is of size N 
            model.train()
            for batch_num, (X,y) in enumerate(train_loader):
                X, y = X.to(device), y.to(device)
                preds, features = model(X)
                optimizer.zero_grad()
                
                ce_loss = cross_entropy_loss(preds, y)
                cl = center_loss(features)
                
                f1, f2, f3 = BHM.get_valid_triples(features, y)
                tl = triplet_loss(f1,f2,f3)
                
                loss = ce_loss + tl + beta * cl
                
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                #register losses
                
            model.eval()
            for batch_num, (X,y) in enumerate(test_loader):
                  X, y = X.to(device), y.to(device)
                  with torch.no_grad():
                        normalized_features = model(X)
                        f1, f2, f3 = BHM.get_valid_triples (normalized_features, y)
                        tl = triplet_loss(f1, f2, f3)
                  
                  
                  
