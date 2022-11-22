import torch
import torch.nn as nn
import torch.nn.functional as F

def sq_l2_dist(x,y):
    """given x,y of shape NxD computes squared L2 distance along column"""
    return ((x-y) ** 2).sum(dim=1)

def cosine_dist(x,y):
    """Given x, y of shape NxD and NxD computes row-wise cosine distance"""
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    
    return (x*y).sum(dim=-1)
    

class TripletLoss(nn.Module):
    def __init__(self, alpha=0.3, use_cosine_dist=False): 
        """The default alpha value is from the Bag-of-Tricks paper. We use Euclidean distance for 
        training following the same paper, but during inference we will use cosine 
        similarity (the center loss compactifies each class to make this feasible)."""
        super().__init__()
        self.alpha = alpha
        self.use_cosine = use_cosine_dist
        
    
    def forward(self, a, p, n):
        """a,p,n are feature matrices of shape NxD"""
        assert len(a.shape) == 2
        dist_fn = cosine_dist if self.use_cosine else sq_l2_dist
        
        dist_positive = dist_fn(a,p)
        dist_negative = dist_fn(a,n)
        
        loss_vector = F.relu(dist_positive - dist_negative + self.alpha)
        num_nonzero = torch.count_nonzero(loss_vector) + 0.000001 #for numeric stability
        
        # from bag of tricks paper, we want to focus on harder cases. 
        return loss_vector.sum() / num_nonzero