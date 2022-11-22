import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.tensorboard as tensorboard
import torch.nn.functional as F
from loguru import logger
import train_tools.BatchHardMining as BHM
from tqdm import tqdm



def train(train_loader, val_loader, model, triplet_loss, center_loss,
          cross_entropy_loss, optimizer, scheduler, train_summary_writer,
          val_summary_writer, device=torch.device("cuda"), num_epochs=1,
          beta=0.0005, should_log=False):
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
    best_val_loss = -1
    for i in (pbar := tqdm(range(num_epochs))):
            if should_log:
                  logger.info(f"Starting train epoch {i}")
            train_one_epoch(
                  train_loader, i, model, triplet_loss, center_loss,
                  cross_entropy_loss, optimizer, train_summary_writer,
                  "Avg Train Loss in each epoch", device, beta, should_log=should_log)
            
            val_loss = val_one_epoch(
                        val_loader, i, model, triplet_loss, val_summary_writer,
                        "Avg Validation/Triplet loss in each epoch"
                        )
            if best_val_loss == -1:
                  best_val_loss = val_loss
            if best_val_loss > val_loss:
                  best_val_loss = val_loss
                  torch.save(model.state_dict(), "../weights/best_model.pth")
      
            train_summary_writer.flush()
            val_summary_writer.flush()
            scheduler.step()
      
                  
                  
                  
def train_one_epoch(
      train_loader, epoch_number, model, triplet_loss, center_loss, cross_entropy_loss, optimizer, summary_writer,  summary_plot_title, 
      device, beta, should_log, eps=1e-12):
      """Run one step of the training loop."""
      log = logger.info
      running_loss = 0.
      model.train()
      num_batches = 0
      
      for batch_num, (X,y) in tqdm(enumerate(train_loader)):
            if should_log: 
                  log(f"in epoch {epoch_number}: batch num = {batch_num}")
            X, y = X.to(device), y.to(device)
            preds, features = model(X)
            
            ce_loss = cross_entropy_loss(preds, y)
            c_loss = center_loss(features, y)
            f1, f2, f3 = BHM.get_valid_triples(features, y, device=device)
            
            #tl = triplet_loss(f1, f2, f3)
            # f1 is N x D, and f2 is also N x D.  
            # We want N x N matrix of distances. 
            sq_positive_dists = ((f1-f2) ** 2).sum(dim=1)
            zero_mask = sq_positive_dists < eps
            non_zero_mask = ~zero_mask
            positive_distance = F.relu(torch.sqrt(non_zero_mask * sq_positive_dists))
            
            sq_negative_dists = ((f1-f3) ** 2).sum(dim=1)
            zero_mask = sq_negative_dists < eps
            non_zero_mask = ~zero_mask
            negative_distance = F.relu(torch.sqrt(non_zero_mask * sq_negative_dists))
            
            tl = triplet_loss(positive_distance - negative_distance, y)
            loss = ce_loss + tl + beta * c_loss
            
            optimizer.zero_grad()
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

            num_batches += 1
      summary_writer.add_scalar(
            summary_plot_title, 
            running_loss / num_batches,
            epoch_number + 1
      )
      
def val_one_epoch(
            val_loader, epoch_number, model, triplet_loss, summary_writer,
            summary_plot_title, device=torch.device("cuda"), eps=1e-12
      ):
      #we use different losses for training and validation since
      #classification loss during validation doesn't have value since we 
      #are using different classes.
      
      running_loss = 0.
      model.eval()
      num_batches = 0
      for batch_num, (X,y) in tqdm(enumerate(val_loader)):
            X, y = X.to(device), y.to(device)
            with torch.no_grad():
                  features = model(X) #during validation use normalized features
                  
                  #ce_loss = cross_entropy_loss(preds, y)
                  #c_loss = center_loss(features)
                  f1, f2, f3 = BHM.get_valid_triples(features, y, device=device)
                  
                  
                  sq_positive_dists = ((f1-f2) ** 2).sum(dim=1)
                  zero_mask = sq_positive_dists < eps
                  non_zero_mask = ~zero_mask
                  positive_distance = F.relu(torch.sqrt(non_zero_mask * sq_positive_dists))
                  
                  sq_negative_dists = ((f1-f3) ** 2).sum(dim=1)
                  zero_mask = sq_negative_dists < eps
                  non_zero_mask = ~zero_mask
                  negative_distance = F.relu(torch.sqrt(non_zero_mask * sq_negative_dists))
            
                  loss = triplet_loss(positive_distance - negative_distance, y)
                  #loss = triplet_loss(f1, f2, f3)
                  
                  #loss = ce_loss + tl + beta * c_loss
                  
                  
                  
                  running_loss += loss.item()
                  num_batches += 1

      summary_writer.add_scalar(
            summary_plot_title, 
            avg_loss := running_loss / num_batches,
            epoch_number + 1
      )
      
      return avg_loss
       