import re
import os
from os.path import basename as bn
from glob import glob
from collections import defaultdict
from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset


from loguru import logger
log = logger.info




def get_market1501_dataset(root="../../data/Market-1501", mode="train", 
                           transform=None, target_transform=None):
    """Mode can be train, query or test and returns the appropriate 
    dataset object."""
    
    assert mode in ["train", "query", "test"]
    
    if mode == "train":
        return Market1501_Train(root, transform, target_transform)
    elif mode == "query":
        return Market1501_Query(root, transform, target_transform)
    else: 
        return Market1501_Test(root, transform, target_transform)
    
    
    
    
    
        



class Market1501_Query(Dataset):
    def __init__(self, root, transform, target_transform):
        super().__init__()
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        
        
        self.dir = os.path.join(self.root, "query")
        
        #get list of filenames.
        self.filenames = sorted(glob(os.path.join(self.dir, "*.jpg")))
        
        regex = re.compile("([0-9]+)_c([0-9])s")
        
        #match the basename of the file 
        #for eg. 0014_c5_s1_001051_00.jpg will give "0014" and "5"
        matches = [regex.match(bn(filename)) for filename in self.filenames]
        
        self.pids, self.camids = zip(*[(int(m.group(1)), int(m.group(2)))
                                        for m in matches])
        #self.pids = torch.tensor(self.pids)
       # self.camids = torch.tensor(self.camids)
        
    def __getitem__(self, idx):
        img = Image.open(self.filenames[idx]).convert("RGB")
        camid = self.camids[idx]
        pid = self.camids[idx]
        
        if self.transform is not None:
            img = self.transform(img)
            
        #img_info = {"img" : img, "camid" : camid, "pid" : pid}
            
        if self.target_transform is not None:
            pid, camid = self.target_transform(pid, camid)
            
        return img, pid, camid
        
        
    def __len__(self):
        return len(self.pids)



class Market1501_Test(Dataset):
    def __init__(self, root, transform, target_transform):
        super().__init__()
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        
        
        self.dir = os.path.join(self.root, "bounding_box_test")
        
        #get list of filenames.
        self.filenames = sorted(glob(os.path.join(self.dir, "*.jpg")))
        
        regex = re.compile("(-?[0-9]+)_c([0-9])s")
        
        #match the basename of the file 
        #for eg. 0014_c5_s1_001051_00.jpg will give "0014" and "5"
        matches = [regex.match(bn(filename)) for filename in self.filenames]
        
        self.pids, self.camids = zip(*[(int(m.group(1)), int(m.group(2)))
                                        for m in matches])
        
        i = 0
        while self.pids[i] == -1:
            i += 1
        
        #remove the "junk" images with id -1, but keep the
        #images with id 0 meant to "distract" the query
        self.filenames = self.filenames[i:]
        self.pids = self.pids[i:]
        self.camids = self.camids[i:]
                
        
    def __getitem__(self, idx):
        img = Image.open(self.filenames[idx]).convert("RGB")
        camid = self.camids[idx]
        pid = self.camids[idx]
        
        if self.transform is not None:
            img = self.transform(img)
            
        #img_info = {"img" : img, "camid" : camid, "pid" : pid}
            
        if self.target_transform is not None:
            pid, camid = self.target_transform(pid, camid)
            
        return img, pid, camid
        
        
    def __len__(self):
        return len(self.pids)


#for this we can just use the older one

class Market1501_Train(Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        self.root = root
        self.label_to_start_stop_idxs = defaultdict(list)
       # self.idx_to_label = defaultdict(lambda : -1)
        #self.labels_to_idxs = defaultdict(list)
        self.labels = []
        self.idx_to_label = {}
        
        self.transform = transform
        self.target_transform = target_transform
        
        
        #self._TEST_FOLDER = os.path.join(self.root, "bounding_box_test")
        #self._QUERY_FOLDER = os.path.join(self.root, "query")
        self._TRAIN_FOLDER = os.path.join(self.root, "bounding_box_train")
        
       
        self.filenames = []
        
        #ignore thumbs.db
        train_filenames = os.listdir(self._TRAIN_FOLDER)[:-1]
        
        latest_label_seen = 2
        self.label_to_start_stop_idxs[latest_label_seen].append(0)
        self.labels = [2]
        for i,filename in enumerate(train_filenames):
            full_filename = os.path.join(self._TRAIN_FOLDER, filename)
            self.filenames.append(full_filename)
            label = int(filename[:4])
            self.idx_to_label[i] = label
            if label > latest_label_seen:
                self.label_to_start_stop_idxs[latest_label_seen].append(i-1)
                latest_label_seen = label
                self.label_to_start_stop_idxs[label] = [i]
                self.labels.append(label)
        #boundary case: the last label won't have an end point
        self.label_to_start_stop_idxs[latest_label_seen].append(len(train_filenames) - 1)
            
        #self.labels_inv[self.labels[i]] = i
        self.labels_inv = {x : i for (i,x) in enumerate(self.labels)}
                    
    
    def __getitem__(self, idx):
        #we will get the image from self.filenames[i]
        img = Image.open(self.filenames[idx]).convert("RGB")
        
        #use labels_inv to get labels between 0 and N-1
        #instead of 1, 4, 7, ...
        label = self.labels_inv[self.idx_to_label[idx]]
        
        if self.transform is not None:
            img = self.transform(img)
            
        if self.target_transform is not None:
            label = self.target_transform(label)
        
        return img, label
    
    def __len__(self):
        return len(self.filenames)
        
        
        