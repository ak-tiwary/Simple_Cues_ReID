import torch
from torch.utils.data import Sampler
from collections import defaultdict
import random


class ClassUniformBatchSampler(Sampler):
    def __init__(self, dataset, P, K, start_indices, num_examples, num_classes):
        """Returns batches of size P x K where P is the number of classes per batch and K is the number of samples per class. Assumes dataset has classes in order and the class lables are from 0 to
        MAX_LABEL. Assumes also that the dataset items with label i are precisely the elements with
        indices start_indices[i] to start_indices[i] + num_examples[i]"""
        self.num_classes = num_classes
        self.start_indices = start_indices
        self.num_examples = num_examples 
        self.dataset = dataset
        self.P = P
        self.K = K
        
        
        
    def __iter__(self):
        """Provides a valid permutation of the indices for the dataset."""
        
        #Shuffle indices of each class in place.
        classwise_shuffled_indices = []
        k_at_time = defaultdict(list) #for each class we want the value to be [[k samples], [k samples], ...]
        
        for i in range(self.num_classes):
            class_batches = []
            start_idx = self.start_indices[i]
            stop_idx = start_idx + self.num_examples[i]
            
            #want a random shuffle of the dataset from start_idx -> start_idx + num_examples
            shuffled_examples = random.sample(range(start_idx, stop_idx), stop_idx - start_idx)
            
            split_batches = list(torch.split(torch.tensor(shuffled_examples), self.K))
            
            #last_batch = n % self.K
            
            #we want last batch to also have size K using random.choices
            split_batches[-1] = torch.tensor(random.choices(shuffled_examples[-(len(shuffled_examples) % self.K) :], k=self.K))
             #we want to pop() elements so the last batch should be at index 0
            
            split_batches = split_batches[::-1]
            k_at_time[i] = [x.tolist() for x in split_batches] #each is a list of k indices.
            
            classwise_shuffled_indices += shuffled_examples
            
        
        
        #We want to extract P classes at a time randomly and pop the k_at_a_time[idx] for each
        #If any class becomes empty, we want to remove it from our set of alive classes.
        #When we have fewer than K classes remaining, we just return what remains.
        
        alive_classes = set(range(self.num_classes))
        
        while len(alive_classes) >= self.P:
            #sample P classes
            
            selected_indices = random.sample(list(alive_classes), k=self.P)    
            x_batch  = []
            
            
            for idx in selected_indices:
                class_batch = k_at_time[idx].pop()
                if not k_at_time[idx]:
                    alive_classes.remove(idx)
                x_batch += class_batch
                
                
            
            random.shuffle(x_batch)
            
            yield x_batch
        
        #if there are fewer than P classes remaining, we will just cycle through each class and add a set of K elements in order
        #until we hit PxK elements. If we are done before we hit PxK elements, we ignore the batch. 
        
        curr_batch = []
        num_items_in_batch = 0
        while len(alive_classes) >= 2: #if there is only one alive class, no point having a triplet loss
            for idx in list(alive_classes):
                class_batch = k_at_time[idx].pop()
                if not k_at_time[idx]:
                    alive_classes.remove(idx)
                
                curr_batch += class_batch
                num_items_in_batch += self.K
            
                if num_items_in_batch == self.P * self.K:
                    random.shuffle(curr_batch)
                    yield curr_batch
                    curr_batch = []
                    num_items_in_batch = 0
                    
        #there are fewer than PxK elements remaining so we ignore them   
        
        return
    
    def __len__(self):
        return len(self.dataset)