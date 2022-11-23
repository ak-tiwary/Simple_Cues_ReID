import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler

import torchvision.transforms as T
import random
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

#from mpl_toolkits.axes_grid1 import ImageGrid
#from loguru import logger
#from tqdm import tqdm


import data_tools.UniformSampler as US
import data_tools.Market1501_New as MKT
import data_tools.RandomErase as RE
import train_tools.CenterLoss as CL
import train_tools.TripletLoss as TL
#import train_tools.BatchHardMining as BHM

import net as N
import train_helper as t



#for reproducibility
random.seed(0)
torch.manual_seed(0)



ROOT = "../data/Market-1501"
P = 16
K = 4
EMBEDDING_DIM = 2048 #from resnet50
STARTING_LR = 0.00035 #from bag of tricks paper
NUM_EPOCHS = 121
SHOULD_LOG = False
DEVICE = torch.device("cuda")
MODEL_SAVE_PATH = "../weights/best_model.pth"
CONTINUE_TRAIN_FLAG = True




def main():
    
    transform = T.Compose([
            T.ToTensor(),
            T.Resize(size=(256,128)),
            T.Pad(padding=10, fill=0),
            T.RandomCrop(size=(256,128)),
            T.RandomErasing()
    ])
    
#     test_transform = T.Compose([T.ToTensor(), T.Resize(size=(256,128))])
    
    
    
    train_set = MKT.get_market1501_dataset(root=ROOT, mode="train", transform=transform)
#     query_set = MKT.get_market1501_dataset(root=ROOT, mode="query", transform=test_transform)
#     test_set = MKT.get_market1501_dataset(root=ROOT, mode="test", transform=test_transform)
    
    # Create train data loader with class balanced batches
    train_labels = train_set.labels
    train_start_indices = {i : train_set.label_to_start_stop_idxs[label][0] 
                    for i,label in enumerate(train_labels)}
    train_num_examples = {i : 
                    (train_set.label_to_start_stop_idxs[label][1] - \
                    train_set.label_to_start_stop_idxs[label][0] + 1)
                    for i, label in enumerate(train_labels)}
    train_num_classes = len(train_labels)
    
    train_k_at = US._get_k_at_time(start_indices=train_start_indices, num_examples=train_num_examples,
                                    K=K, num_classes=train_num_classes)
    
    train_sampler = US.ClassUniformBatchSampler(
            dataset=train_set, P=P, K=K, k_at_time=train_k_at, num_classes=train_num_classes,
            start_indices=train_start_indices, num_examples=train_num_examples
    )
    
    train_loader = DataLoader(dataset=train_set, batch_sampler=train_sampler, 
                              num_workers=2, pin_memory=True)
    
    
#     #query and gallery loaders are as is
#     query_loader = DataLoader(query_set, batch_size=64, shuffle=False, drop_last=False, pin_memory=False)
#     gallery_loader = DataLoader(test_set, batch_size=64, shuffle=False, drop_last=False, pin_memory=False)
    
    # want normalized feature vectors during inference
    model = N.Net(num_classes=train_num_classes, device=DEVICE, normalize=True)
    
    
    
    triplet_loss = TL.TripletLoss()
    center_loss = CL.CenterLoss(num_classes=train_num_classes, embedding_dim=EMBEDDING_DIM).to(DEVICE)
    cross_entropy_loss = nn.CrossEntropyLoss(label_smoothing=0.1) #from BagOfTricks paper
    
    
    optimizer = torch.optim.Adam(model.parameters(), lr=STARTING_LR, weight_decay=5e-4)
    scheduler1 = lr_scheduler.LambdaLR(optimizer, lr_lambda= lambda epoch: epoch / 10)
    scheduler2 = lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)
    #scheduler3_prime = lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=5) #depends on num epochs
    #scheduler2 = lr_scheduler.ChainedScheduler([scheduler2_prime, scheduler3_prime])
    scheduler3 = lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.4)
    scheduler = lr_scheduler.SequentialLR(optimizer,
                                            [scheduler1, scheduler2, scheduler3],
                                                milestones=[10, 71])
    
    train_writer = SummaryWriter("runs/metric_learning_7")
#     mAP_writer = SummaryWriter("runs/metric_learning_4")
#     accuracy_writer = SummaryWriter("runs/metric_learning_5")
    
    
    continue_from_epoch=None
    ###################################
#     Load from file to contine training
    if CONTINUE_TRAIN_FLAG:
        save_dict = torch.load(MODEL_SAVE_PATH)
        model.load_state_dict(save_dict["model_state_dict"])
        optimizer.load_state_dict(save_dict["optimizer_state_dict"])
        optimizer.param_groups[0]["lr"] = 0.0000015 ###CHANGE
        loss = save_dict["loss"]
        START_EPOCH=71 #change this and the scheduler
       #scheduler=lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=8)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.4)
        continue_from_epoch=(START_EPOCH,loss)
    #####################################
    
    t.train(train_loader, model, triplet_loss, center_loss, cross_entropy_loss,
            optimizer, scheduler, train_writer, DEVICE, NUM_EPOCHS, should_log=SHOULD_LOG, continue_from_epoch=continue_from_epoch)
      
      







if __name__ == "__main__":
    main()


