import torch
import torch.nn as nn
import torch.nn.functional as F

def sq_l2_dist(x,y):
    """given x,y of shape NxD computes squared L2 distance along column"""
    return ((x-y) ** 2).sum(dim=1)
    

class TripletLoss(nn.Module):
    def __init__(self, margin, alpha=0.3): #alpha from Bag of Tricks paper
        super().__init__()
        self.alpha = alpha
        
    
    def forward(self, a, p, n):
        """a,p,n are feature matrices of shape NxD"""
        dist_positive = sq_l2_dist(a,p)
        dist_negative = sq_l2_dist(a,n)
        
        return F.relu(dist_positive - dist_negative + self.alpha).mean()