import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.tensorboard as tensorborad

import train_tools.BatchHardMining as BHM
from tqdm import tqdm



def train(train_loader, val_loader, model, triplet_loss, center_loss,
          cross_entropy_loss, optimizer, scheduler, train_summary_writer,
          val_summary_writer, device=torch.device("cuda"), num_epochs=1,
          beta=0.0005):
    """model is the network. scheduler is the learning rate scheduler,
    summary_writer is the tensorboard summary writer.
    
    Beta is a hyperparameter, set to 0.0005 from the Bag of Tricks paper.
    
    We use different train and val SummaryWriters since we calculate 
    different losses during train and validation times. When training
    we want our model to also use the cross entropy and center as well
    (the latter is used to compactify each class). During validation
    we only want to use the triplet loss so that we are evaluating 
    what we intend the model to be used for: providing a similarity
    measure."""
    
    #has a learnable parameter
    center_loss = center_loss.to(device)
    
    for i in (pbar := tqdm(num_epochs)):
          train_one_epoch(
                train_loader, i, model, triplet_loss, center_loss,
                cross_entropy_loss, optimizer, train_summary_writer,
                "Avg Train Loss in each epoch", device, beta)
          
          val_one_epoch(
                val_loader, i, model, triplet_loss, val_summary_writer,
                "Avg Validation/Triplet loss in each epoch")
          
          scheduler.step()
      
                  
                  
                  
def train_one_epoch(
      train_loader, epoch_number, model, triplet_loss, center_loss, cross_entropy_loss, optimizer, summary_writer,  summary_plot_title, 
      device, beta):
      """Run one step of the training loop."""
      running_loss = 0.
      model.train()
      num_batches = len(train_loader)
      for batch_num, (X,y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)
            preds, features = model(X)
            
            ce_loss = cross_entropy_loss(preds, y)
            c_loss = center_loss(features)
            f1, f2, f3 = BHM.get_valid_triples(features, y)
            
            tl = triplet_loss(f1, f2, f3)
            
            loss = ce_loss + tl + beta * c_loss
            
            optimizer.zero_grad()
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

      
      summary_writer.add_scalar(
            summary_plot_title, 
            {"training_loss" : running_loss / num_batches},
            epoch_number + 1
      )
      
def val_one_epoch(
            val_loader, epoch_number, model, triplet_loss, summary_writer,
            summary_plot_title, device=torch.device("cuda")
      ):
      #we use different losses for training and validation since
      #classification loss during validation doesn't have value since we 
      #are using different classes.
      
      running_loss = 0.
      model.eval()
      num_batches = len(val_loader)
      for batch_num, (X,y) in enumerate(val_loader):
            X, y = X.to(device), y.to(device)
            with torch.no_grad():
                  features = model(X) #during validation use normalized features
                  
                  #ce_loss = cross_entropy_loss(preds, y)
                  #c_loss = center_loss(features)
                  f1, f2, f3 = BHM.get_valid_triples(features, y)
                  
                  loss = triplet_loss(f1, f2, f3)
                  
                  #loss = ce_loss + tl + beta * c_loss
                  
                  
                  
                  running_loss += loss.item()

      
      summary_writer.add_scalar(
            summary_plot_title, 
            {"val_loss" : running_loss / num_batches},
            epoch_number + 1
      )
       