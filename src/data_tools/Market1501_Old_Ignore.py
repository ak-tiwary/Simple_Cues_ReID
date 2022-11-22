import os
from collections import defaultdict
from PIL import Image
from torch.utils.data import Dataset



class Market1501(Dataset):
    def __init__(self, root, train=True,
                 transform=None, target_transform=None):
        self.are_we_training = train
        self.root = root
        self.label_to_start_stop_idxs = defaultdict(list)
       # self.idx_to_label = defaultdict(lambda : -1)
        #self.labels_to_idxs = defaultdict(list)
        self.labels = []
        self.idx_to_label = {}
        
        self.transform = transform
        self.target_transform = target_transform
        
        
        self._TEST_FOLDER = os.path.join(self.root, "bounding_box_test")
        self._QUERY_FOLDER = os.path.join(self.root, "query")
        self._TRAIN_FOLDER = os.path.join(self.root, "bounding_box_train")
        
        if self.are_we_training:
            
            self.filenames = []
            #self.curr_idx = 0
            test_box_filenames = os.listdir(self._TEST_FOLDER)
            query_filenames = os.listdir(self._QUERY_FOLDER)
            
            
            #ignore junk/distraction files
            test_box_filenames = test_box_filenames[6617:] 
            
            #ignore thumbs.db
            test_box_filenames = test_box_filenames[:-1]
            query_filenames = query_filenames[:-1]
            
            self.label_to_filenames = defaultdict(list) #label -> list of filenames
            
            #we have label_to_filename[label] = [filename, filename, ...]
            latest_label_seen = -1
            
            #we want to have all the elements with the same label occur together
            #so we fiddle around a bit to make that happen
            for filename in test_box_filenames:
                full_filename = os.path.join(self._TEST_FOLDER, filename)
                label = int(filename[:4])
                self.label_to_filenames[label].append(full_filename)
                if label > latest_label_seen: #new label
                    self.labels.append(label)
                    latest_label_seen = label
                
            for filename in query_filenames:
                full_filename = os.path.join(self._QUERY_FOLDER, filename)
                label = int(filename[:4])
                self.label_to_filenames[label].append(full_filename)
                
            idx = 0
            for label in self.labels:
                start_idx = idx 
                for filename in self.label_to_filenames[label]:
                    self.filenames.append(filename)
                    self.idx_to_label[idx] = label
                    idx += 1
                end_idx = idx - 1
                self.label_to_start_stop_idxs[label] = [start_idx, end_idx]
            
        else:
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