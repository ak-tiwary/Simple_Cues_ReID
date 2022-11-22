import numpy as np


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T


import random
import math


class RandomErase(nn.Module):
    def __init__(self, p_erase=0.5, aspect_ratio_range=(0.3,3.33), area_range=(0.02,0.4)):
        """The a portion of the image is randomly erased with probability p_erase. 
        The aspect ratio of the erased rectangle lies between the range provided by 
        aspect_ratio_range. The ratio of the area of the rectangle to the area of the
        whole rectangle lies within the area_range.
        
        For more details see the "Bag of Tricks" 2019 paper by Luo et. al."""
        super().__init__()
        self.p = p_erase
        self.r1, self.r2 = aspect_ratio_range
        self.a1, self.a2 = area_range
        
    def forward(self, img):
        assert len(img.shape) == 3
        C, H, W = img.shape
        
        
        if random.random() < 0.5:
            #perform random erasing
            S = H * W
            
            r_e = random.uniform(self.r1, self.r2)
            S_e = S * random.uniform(self.a1, self.a2)
            
            #r = w/h, S = w*h
            w_e = math.sqrt(r_e * S_e)
            h_e = w_e / r_e
            
            w_e, h_e = int(w_e), int(h_e)
            #X goes from 0 to W-1
            #Y goes from 0 to H-1
            
            
            #we want no biasing so we want to randomly pick the center. So we require
            # x_e >= w_e / 2, y_e >= h_e / 2
            # and x_e + w_e / 2 <= W-1, y_e + h_e / 2 <= H-1
            # so ceil(w_e / 2) <= x_e <= floor(W-1 - w_e/2)
            x_e = random.randrange(math.ceil(w_e / 2), math.floor(W - (w_e / 2) - 1))
            y_e = random.randrange(math.ceil(h_e / 2), math.floor(H - (h_e / 2) - 1))
            
            
            x_top_left = math.floor(x_e - (w_e / 2))
            y_top_left = math.ceil(y_e - (h_e / 2))
            
            
            
            #just set the box to grey
            grey = torch.tensor([122, 122, 122]).reshape((-1,1,1))
            img[:, y_top_left : y_top_left + h_e, x_top_left : x_top_left + w_e] = grey
            
            
            return img
        else:
            return img