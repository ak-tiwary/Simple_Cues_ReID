#This file evaluates the model using the mAP and rank1 metrics.
import numpy as np

from scipy.io import loadmat

import torch
import torch.nn.functional as F
from tqdm import tqdm

from loguru import logger
log = logger.info
from net import Net



WEIGHTS_PATH = "./weights/best_model.pth"
NUM_CLASSES = 751 #from Market1501
DEVICE = torch.device("cuda")
ROOT = "../data/Market-1501"
MATRIX_FILEPATH = "../result.mat"
EMBEDDING_DIM = 2048










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
                         gallery_camids):
    """Compute mAP and Rank-1 Accuracy metrics. All inputs are np arrays"""
    APs = np.zeros_like(query_pids, dtype=float)
    #APs = torch.zeroslike(query_pids).to(torch.float)
    correct_ids = np.zeros_like(query_pids, dtype=float)
    #correct_ids = torch.zeroslike(query_pids).to(torch.float)
    
    #at row i we have the indices of the galleries sorted by proximity
    #query_to_gallery_indices = torch.argsort(distance_matrix, dim=-1)
    query_to_gallery_indices = np.argsort(distance_matrix, axis=-1)
    
    for idx, query_pid, query_camid in (p:=tqdm(enumerate(zip(query_pids, query_camids)))):
        p.set_description(f"Evaluating query {idx}")
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
        
        correct_matches = correct_matches.astype(float)
        #to calculate AP we need to sum the cumulative precision
        #at those indices where the pid is correct and divide by num_gallery
        
        #the formula for precision upto index k is:
        # 
        # number of correct up to k
        # --------------------------
        # number of entries upto k
        
        #we can calculate the numerator by using torch.cumsum
        
        precision_num = np.cumsum(correct_matches, axis=0)
        precision_denom = np.arange(1, num_gallery).astype(float)
        # precision_num = torch.cumsum(correct_matches, dim=0)
        # precision_denom = torch.arange(1,num_gallery+1).to(device, torch.float)
        
        precision = precision_num / precision_denom
        
        temp = precision * correct_matches
        APs[idx] = temp.sum() / num_gallery
        
        correct_ids[idx] = correct_matches[0]
    
    #num_queries = len(query_pids)
    
    mAP = APs.mean()
    accuracy = correct_ids.mean()
    
    return mAP, accuracy

    
    
def evaluate():
    """will read the feature vectors from saved matrix and calculate mAP and rank 1 accuracy."""
    result = loadmat(MATRIX_FILEPATH)
    q_f = torch.FloatTensor(result["query_f"]).cuda()
    q_id = result["query_ids"][0] #they are saved as 2d matrices by scipy
    q_camid = result["query_camids"][0]
    
    g_f = torch.FloatTensor(result["gallery_f"]).cuda()
    g_id = result["gallery_ids"][0] #they are saved as 2d matrices by scipy
    g_camid = result["gallery_camids"][0]
    
    log(f"Starting distance matrix calculation.")
    dist_matrix = compute_cosine_dist_matrix(q_f, g_f, already_normalized=True)
    log(f"Distance matrix calculated. Has shape {dist_matrix.shape}.")
    dist_matrix = dist_matrix.cpu().numpy()
    log(f"Converted distance matrix to numpy.")
    
    return compute_eval_metrics(dist_matrix, q_id, q_camid, g_id, g_camid)
        


if __name__ == "main":
   
    
    state = torch.load(WEIGHTS_PATH)
    
    # model = Net(num_classes=751)
    # model.load_state_dict(state["model_state_dict"])
    # log(f"model loaded")
    mAP, rank1 = evaluate()
    log(f"Evaluation complete. Rank 1 accuracy is {rank1} and mAP is {mAP}")    
    #print(f"The mAP is {mAP} and the rank 1 accuracy is {rank1}")
    