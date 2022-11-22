#This file evaluates the model using the mAP and rank1 metrics.
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T


from loguru import logger
log = logger.info


def normalize(features):
    """Given an NxD matrix performs L2 normalization of features."""
    return F.normalize(features, p=2, dim=1)

def compute_cosine_dist_matrix(query_features, gallery_features, already_normalized=False):
    """Given query and gallery features of shape NxD and MxD, computes
    a matrix of cosine distances so that d(i,j) = 1 - cos(f_i,f_j).
    
    If already_normalized is False then the vectors will be L2 normalized.
    """
    
    if not already_normalized:
        query_features = normalize(query_features)
        gallery_features = normalize(gallery_features)
        
        
    return 1-query_features @ gallery_features.T


def compute_eval_metrics(distance_matrix, query_pids, query_camids, gallery_pids, 
                         gallery_camids, device=torch.device("cuda")):
    """Compute mAP and Rank-1 Accuracy metrics."""
    APs = torch.zeroslike(query_pids).to(torch.float)
    correct_ids = torch.zeroslike(query_pids).to(torch.float)
    
    #at row i we have the indices of the galleries sorted by proximity
    query_to_gallery_indices = torch.argsort(distance_matrix, dim=-1)
    
    for idx, query_pid, query_camid in enumerate(zip(query_pids, query_camids)):
        gallery_indices = query_to_gallery_indices[idx]
        gallery_pids_ordered = gallery_pids[gallery_indices]
        gallery_camids_ordered = gallery_camids[gallery_indices]
        #we want to remove all the indices with the same pid and cam id as the query
        same_pid_camid_mask = (gallery_pids_ordered == query_pid) & \
                              (gallery_camids_ordered == query_camid)
        
        keep_indices = ~same_pid_camid_mask
        
        #only consider these indices for gallery
        gallery_indices = gallery_indices[keep_indices]
        gallery_pids_ordered = gallery_pids[gallery_indices]
        #gallery_camids_ordered = gallery_camids[gallery_indices]
        correct_matches = gallery_pids_ordered == query_pid
        
        num_gallery = len(gallery_indices)
        
        assert num_gallery > 0, f"query id = {query_pid} not in gallery"
        
        correct_matches = correct_matches.to(torch.float)
        #to calculate AP we need to sum the cumulative precision
        #at those indices where the pid is correct and divide by num_gallery
        
        #the formula for precision upto index k is:
        # 
        # number of correct up to k
        # --------------------------
        # number of entries upto k
        
        #we can calculate the numerator by using torch.cumsum
        
        precision_num = torch.cumsum(correct_matches, dim=0)
        precision_denom = torch.arange(1,num_gallery+1).to(device, torch.float)
        
        precision = precision_num / precision_denom
        
        temp = (precision * correct_matches.to(torch.float))
        APs[idx] = temp.sum() / num_gallery
        
        correct_ids[idx] = correct_matches[0]
    
    #num_queries = len(query_pids)
    
    mAP = APs.mean().item()
    accuracy = torch.mean(correct_ids)
    
    return mAP, accuracy

    
        
    