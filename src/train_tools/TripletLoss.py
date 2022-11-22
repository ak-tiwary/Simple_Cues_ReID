import torch
import torch.nn as nn
import torch.nn.functional as F

def sq_l2_dist(x,y):
    """given x,y of shape NxD computes squared L2 distance along column"""
    return ((x-y) ** 2).sum(dim=1)
    

class TripletLoss(nn.Module):
    def __init__(self, alpha=0.3): #alpha from Bag of Tricks paper
        super().__init__()
        self.alpha = alpha
        
    
    def forward(self, a, p, n):
        """a,p,n are feature matrices of shape NxD"""
        assert len(a.shape) == 2
        dist_positive = sq_l2_dist(a,p)
        dist_negative = sq_l2_dist(a,n)
        
        loss_vector = F.relu(dist_positive - dist_negative + self.alpha)
        num_nonzero = torch.count_nonzero(loss_vector) + 0.000001 #for numeric stability
        
        # from bag of tricks paper, we want to focus on harder cases. 
        return loss_vector.sum() / num_nonzero