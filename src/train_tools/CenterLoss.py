import torch
import torch.nn as nn



class CenterLoss(nn.Module):
    def __init__(self, num_classes, embedding_dim):
        """Assume that the classes are 0 to num_classes-1 as usual."""
        self.centers = nn.Parameter(torch.empty((num_classes, embedding_dim), dtype=torch.float))
        nn.init.xavier_normal_(self.centers)
        self.mse = nn.MSELoss()
        
    def forward(self, features, class_labels):
        """Assumes features is Nxd and class labels is N"""
        centers = self.centers[class_labels] #so centers[j] is the center associated with label j
        return self.mse(features, centers)
        