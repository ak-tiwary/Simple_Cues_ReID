import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T


import random




def compute_distance_matrix(features, distance_fn="L2", eps=1e-12):
    """Assumes features is an NxD matrix. Computes NxN pairwise distance matrix between the rows and returns it.
    Currently assumes that the distance function is the L2 distance function as the method below only works
    for this function. Alternative functions might require longer computation since easy vectorization may
    not be possible."""
    
    #To compute the distance matrix we use |a-b|^2 = |a|^2 - 2 a.b + |b|^2
    X = features @ features.T #X has shape NxN and X[i,j] = f_i . f_j
    sq_norm_features = torch.diagonal(X) #length N, f_i . f_i
    squared_dist = sq_norm_features.unsqueeze(1) - 2 * X + sq_norm_features.unsqueeze(0)
    
    #we need to ensure positivity before taking square root due to floating point errors
    squared_dist = F.relu(squared_dist)
    
    #for numerical stability we add epsilon whereever squared_dist is 0 before taking sqrt
    #this helps during backprop because the derivative of sqrt(x) is 1/2sqrt(x). 
    #See https://www.robots.ox.ac.uk/~albanie/notes/Euclidean_distance_trick.pdf
    
    zero_mask = squared_dist == 0.
    
    
    squared_dist += zero_mask.to(torch.float) * eps #whereever it is zero, add epsilon
    
    distance_matrix = torch.sqrt(squared_dist)
    distance_matrix[zero_mask] = 0.
    
    return distance_matrix
    




def get_triple_indices(distance_matrix, labels):
    """Given an NxN distance matrix between and a N label array returns a tensor of length Mx3 with i,j,k being valid triples
    such that for any i, (i,j) is the furthest apart positive example and (i,k) is the closest negative example.
    
    The sampling strategy ensures that every batch has at least K elements in one class and at least P diff. classes."""
    #dist_matrix = distance_matrix.copy() #don't want changes affected
    N = distance_matrix.shape[0]
    
    
    #we want a mask such that mask[i,j] = True when
    #i and j are the same class
    
    same_class_mask = labels.unsqueeze(0) == labels.unsqueeze(1)
    diff_class_mask = ~same_class_mask
    
    #since we want to pick the farthest index in the same class
    #we use a mask to set the distance of all the (i,j) with diff
    #classes to -1. That way such indices will not affect the
    #maximum selection.
    
    #is 1+d[i,j] whenever i,j are diff class. else 0
    temp_matrix = diff_class_mask * (1 + distance_matrix)
    
    #d[i,j] when i,j same class, otherwise -1
    modified_dist_matrix = distance_matrix - temp_matrix
    
    _, farthest = torch.max(modified_dist_matrix, dim=1) #shape is Nx1 and furthest[i] = j if fj is furthest from fi
    
    
    
    M = torch.max(distance_matrix) + 1
    
    #to get the closest distance in a different class
    #we set d[i,j] = M if i,j in the same class
    
    #is M-d[i,j] if i,j  same class else 0
    temp_matrix = same_class_mask * (M - distance_matrix)
    
    #M is i,j same class else d[i,j]
    modified_dist_matrix = distance_matrix + temp_matrix
    
    _, closest = torch.min(modified_dist_matrix, dim=1) 
    
    
    
    return torch.stack([torch.arange(N), farthest, closest], dim=1)
    

    
    
def get_valid_triples(features, labels):
    """Given an Nxd matrix of N features and corresponding N labels, returns f1, f2, f3, each of shape Nxd, giving
    anchors, positives, and negatives."""
    
    dist_matrix = compute_distance_matrix(features)
    
    indices = get_triple_indices(dist_matrix, labels)
    ind1, ind2, ind3 = indices.T
    
    return features[ind1], features[ind2], features[ind3]
    