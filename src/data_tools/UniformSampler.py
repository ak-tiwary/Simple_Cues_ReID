import torch
from torch.utils.data import Sampler
from collections import defaultdict
import random
from copy import deepcopy

def _get_k_at_time(start_indices, num_examples, K, num_classes):
    #Shuffle indices of each class in place.
    #classwise_shuffled_indices = []
    k_at_time = defaultdict(list) #for each class we want the value to be [[k samples], [k samples], ...]
        #log(f"Before for loop in __iter__, num_classes = {num_classes}")
    for i in range(num_classes):
        class_batches = []
        start_idx = start_indices[i]
        stop_idx = start_idx + num_examples[i]
            #log(f"For class {i}, start_idx = {start_idx} and stop_idx = {stop_idx}")
            #want a random shuffle of the dataset from start_idx -> start_idx + num_examples
        shuffled_examples = random.sample(range(start_idx, stop_idx), stop_idx - start_idx)
            #log(f"For class {i}, shuffled_examples = {shuffled_examples}")
        split_batches = list(torch.split(torch.tensor(shuffled_examples), K))
            #log(f"For class {i}, split_batches start = {split_batches}")
            #last_batch = n % self.K
            
            #we want last batch to also have size K using random.choices
        split_batches[-1] = torch.tensor(random.choices(shuffled_examples[-(len(shuffled_examples) % K) :], k=K))
             #we want to pop() elements so the last batch should be at index 0
            #log(f"For class {i}, split_batches after resizing last element = {split_batches}")
        split_batches = split_batches[::-1]
        k_at_time[i] = [x.tolist() for x in split_batches] #each is a list of k indices.
            #log(f"For class {i}, k_at_time[i] = {k_at_time[i]}")
            #classwise_shuffled_indices += shuffled_examples
            
        #log(f"k_at_a_time = {k_at_time}")
    #log("k_at_time created.")
    return k_at_time

class ClassUniformBatchSampler(Sampler):
    """Returns the classes in order, but with the entries of each class shuffled."""
    def __init__(self, dataset, P, K, k_at_time, num_classes, 
                 start_indices, num_examples, logger=None):
        """Returns batches of size P x K where P is the number of classes per batch and K is the number of samples per class."""
        self.num_classes = num_classes
        self.start_indices = start_indices
        self.num_examples = num_examples 
        self.dataset = dataset
        self.P = P
        self.K = K
        self.k_at_time = k_at_time
        
        if logger is not None:
            logger.info("initialization of sampler complete.")
            
        self.logger = logger
        
        
        
    def __iter__(self):
        """Provides a valid permutation of the indices for the dataset."""
        if self.logger is not None:
            self.logger.info("__iter__ called")
        k_at_time = deepcopy(self.k_at_time)
        
        #We want to extract P classes at a time randomly and pop the k_at_a_time[idx] for each
        #If any class becomes empty, we want to remove it from our set of alive classes.
        #When we have fewer than K classes remaining, we just return what remains.
        
        alive_classes = set(range(self.num_classes))
        i = 1
        while len(alive_classes) >= self.P:
            #sample P classes
            if self.logger is not None:
                self.logger.info(f"In step {i}: ---------------------------")
            selected_indices = random.sample(list(alive_classes), k=self.P)    
            x_batch  = []
            
            
            for idx in selected_indices:
                class_batch = k_at_time[idx].pop()
                if not k_at_time[idx]:
                    alive_classes.remove(idx)
                x_batch += class_batch
            if self.logger is not None:
                self.logger.info(f"indexes selected")    
            
            random.shuffle(x_batch)
            if self.logger is not None:
                self.logger.info(f"yielding batch {i}")
            i += 1
            yield x_batch
        
        #if there are fewer than P classes remaining, we will just cycle through each class and add a set of K elements in order
        #until we hit PxK elements. If we are done before we hit PxK elements, we ignore the batch. 
        
        curr_batch = []
        num_items_in_batch = 0
        if self.logger is not None:
            self.logger.info(f"now in second part")
        while len(alive_classes) >= 2: #if there is only one alive class, no point having a triplet loss
            for idx in list(alive_classes):
                class_batch = k_at_time[idx].pop()
                if not k_at_time[idx]:
                    alive_classes.remove(idx)
                
                curr_batch += class_batch
                num_items_in_batch += self.K
            
                if num_items_in_batch == self.P * self.K:
                    random.shuffle(curr_batch)
                    if self.logger is not None:
                        self.logger.info(f"yielding batch {i}")
                    i += 1
                    yield curr_batch
                    curr_batch = []
                    num_items_in_batch = 0
                    
        #there are fewer than PxK elements remaining    
        
        return
    
    def __len__(self):
        if self.logger is not None:
            self.logger.info(f"length computed. lendth = {len(self.dataset)}")
        return len(self.dataset)

