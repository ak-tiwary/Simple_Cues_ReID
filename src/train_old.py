import gc

import torch
import torch.nn.functional as F
from loguru import logger
import train_tools.BatchHardMining as BHM
from tqdm import tqdm
from evaluate import compute_cosine_dist_matrix as get_dist_matrix
from evaluate import compute_eval_metrics as _evaluate


def train(train_loader, query_loader, gallery_loader, model, triplet_loss, center_loss,
          cross_entropy_loss, optimizer, scheduler, train_summary_writer,
          eval_summary_writers, device=torch.device("cuda"), num_epochs=1,
          beta=0.0005, should_log=False, mAP_loaded=-1, acc_loaded=-1):
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
    best_accuracy = acc_loaded
    best_mAP = mAP_loaded
    for i in (pbar := tqdm(range(num_epochs))):
            # if should_log:
            #       logger.info(f"Starting train epoch {i}")
            train_one_epoch(train_loader, i, model, triplet_loss, center_loss, 
                            cross_entropy_loss, optimizer, train_summary_writer,
                            "Avg Train Loss in each epoch", device, beta, 
                            should_log=should_log)
            
            if i%5 == 0: #every fifth epoch, since evaluating is costly
                
                gc.collect()
                torch.cuda.empty_cache()
                
                #we evaluate on the CPU because of size issues
                mAP, rank1 = evaluate(query_loader, gallery_loader, i, model.to("cpu"),
                                      eval_summary_writers, ("mAP", "Accuracy"))
                
                if mAP > best_mAP and rank1 > best_accuracy:
                    best_mAP = mAP
                    best_accuracy = rank1
                    torch.save({
                        "model_state_dict" : model.state_dict(),
                        "optimizer_state_dict" : optimizer.state_dict(),
                        "mAP" : mAP,
                        "rank1" : rank1,
                    }, "../weights/best_model.pth")

            train_summary_writer.flush()
            
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
            running_loss / num_batches,
            epoch_number + 1
      )
      
def evaluate(query_loader, gallery_loader, epoch_number, model, summary_writers,
            summary_plot_titles):
        """summary_writers is a tuple of SummaryWriters for mAP and for Rank-1 accuracy.
        
        The matrix computed is huge so we work on CPU"""
        #we use different losses for training and validation since
        #classification loss during validation doesn't have value since we 
        #are using different classes.
        
        #device=torch.device("cuda")
        
      
        mAP_writer, rank1_writer = summary_writers
        mAP_title, rank1_title = summary_plot_titles
      
        
        # model.imagenet_mean = model.imagenet_mean.to("cpu")
        # model.imagenet_sd = model.imagenet_sd.to("cpu")
        
        model.eval()
        
        #get query features and cam ids
        all_query_features = []
        all_query_pids = []
        all_query_camids = []
        logger.info(f"")
        for queries, query_pids, query_camids in query_loader:
            queries = queries
            query_pids = query_pids
            query_camids = query_camids
            
            query_features = model(queries)
            all_query_features.append(query_features)
            all_query_pids.append(query_pids)
            all_query_camids.append(query_camids)
            
        query_features = torch.stack(all_query_features, dim=0)
        query_pids = torch.stack(all_query_pids, dim=0)
        query_camids = torch.stack(all_query_camids, dim=0)
        
        
        all_gallery_features = []
        all_gallery_pids = []
        all_gallery_camids = []
        for galleries, gallery_pids, gallery_camids in gallery_loader:
            galleries = galleries
            gallery_pids = gallery_pids
            gallery_camids = gallery_camids
            
            gallery_features = model(galleries)
            all_gallery_features.append(query_features)
            all_gallery_pids.append(gallery_pids)
            all_gallery_camids.append(gallery_camids)
            
        gallery_features = torch.stack(all_gallery_features, dim=0)
        gallery_pids = torch.stack(all_gallery_pids, dim=0)
        gallery_camids = torch.stack(all_gallery_camids, dim=0)
            
            
        
        
        dist_matrix = get_dist_matrix(query_features, gallery_features)
        mAP, rank1 = _evaluate(dist_matrix, query_pids, query_camids,
                               gallery_pids, gallery_camids, device=torch.device("cpu"))

       
        mAP_writer.add_scalar(mAP_title, mAP, epoch_number + 1)
        rank1_writer.add_scalar(rank1_title, rank1, epoch_number + 1)
        
        
        mAP_writer.flush()
        rank1_writer.flush()
        
        return mAP, rank1

        