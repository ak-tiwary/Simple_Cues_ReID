import torch
from loguru import logger
import train_tools.BatchHardMining as BHM
from tqdm import tqdm



def train(train_loader, model, triplet_loss, center_loss,
          cross_entropy_loss, optimizer, scheduler, train_summary_writer, device=torch.device("cuda"), 
          num_epochs=1, beta=0.0005, should_log=False,
          continue_from_epoch=None):
    """model is the network. scheduler is the learning rate scheduler,
    summary_writer is the tensorboard summary writer.
    
    Beta is a hyperparameter, set to 0.0005 from the Bag of Tricks paper.
    
    if continue_from_epoch is not None, will start training from that point
    """
    
    #has a learnable parameter
    center_loss = center_loss.to(device)
    best_loss = 10000000 if continue_from_epoch is None else continue_from_epoch[1]
    loss = best_loss
    for i in (pbar := tqdm(range(num_epochs))):
            pbar.set_description(f"Loss: {loss:.2f}, i : {i}")
            # if should_log:
            #       logger.info(f"Starting train epoch {i}")
            if continue_from_epoch is not None and i < continue_from_epoch[0]:
                  logger.info(f"continuing {i}")
                  continue
            loss = train_one_epoch(train_loader, i, model, triplet_loss, center_loss, 
                            cross_entropy_loss, optimizer, train_summary_writer,
                            "Avg Train Loss in each epoch", device, beta, 
                            should_log=should_log)
            
            if i % 5 == 0:
                  if loss < best_loss:
                        torch.save({
                                          "model_state_dict" : model.state_dict(),
                                          "optimizer_state_dict" : optimizer.state_dict(),
                                          "loss" : loss,
                                    }, 
                                   "../weights/best_model.pth")

            train_summary_writer.flush()
            
            # if i == 71:
            #       #scheduler._last_lr = optimizer.param_groups[0]['lr']
            #       scheduler._last_lr = 0.0000035
            
            # scheduler.step() if i <= 70 else scheduler.step(loss)
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
            
            tl = triplet_loss(f1, f2, f3)
            
            loss = ce_loss + tl + beta * c_loss
            
            optimizer.zero_grad()
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

            num_batches += 1
      summary_writer.add_scalar(
            summary_plot_title, 
            avg_loss := (running_loss / num_batches),
            epoch_number + 1
      )
      
      return avg_loss
  