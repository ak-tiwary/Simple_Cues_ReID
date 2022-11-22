import re
import os
from os.path import basename as bn
from glob import glob
from collections import defaultdict
from PIL import Image
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
                
        
    def __getitem__(self, idx):
        img = Image.open(self.filenames[idx]).convert("RGB")
        camid = self.camids[idx]
        pid = self.camids[idx]
        
        if self.transform is not None:
            img = self.transform(img)
            
        img_info = {"img" : img, "camid" : camid, "pid" : pid}
            
        if self.target_transform is not None:
            img_info = self.target_transform(img_info)
            
        return img_info
        
        
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
            
        img_info = {"img" : img, "camid" : camid, "pid" : pid}
            
        if self.target_transform is not None:
            img_info = self.target_transform(img_info)
            
        return img_info
        
        
    def __len__(self):
        return len(self.pids)


#for this we can just use the older one
class Market1501_Train(Dataset):
    def __init__(self, root, 
                 transform, target_transform):
        raise NotImplementedError()

        
        
        
        
        