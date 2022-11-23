#This file is meant to be run only before evaluation.
#will identify feature matrices for query and gallery and save them, along with pids and camids

#This file is heavily based on the test.py file from https://github.com/layumi/Person-reid-triple-loss

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader


from tqdm import tqdm
from scipy.io import savemat
from net import Net
from data_tools.Market1501_New import get_market1501_dataset

from loguru import logger
log = logger.info

WEIGHTS_PATH = "../weights/best_model.pth" 
NUM_CLASSES = 751 #from Market1501
DEVICE = torch.device("cpu")
ROOT = "../data/Market-1501"
MATRIX_FILEPATH = "../result.mat"
EMBEDDING_DIM = 2048

def get_query_test_loaders(transform):
    query_set = get_market1501_dataset(root=ROOT, mode="query", transform=transform)
    test_set = get_market1501_dataset(root=ROOT, mode="test", transform=transform)
    return query_set, test_set

def save_feature_matrix(model, query_loader, gallery_loader, device=DEVICE):
    model.eval()
    all_q_features = torch.empty((0,EMBEDDING_DIM), dtype=torch.float, device=device)
    all_qids = torch.empty((0,), dtype=torch.int)
    all_qcamids = torch.empty((0,), dtype=torch.int)
    
    
    for i,(q, q_id, q_camid) in (pbar := tqdm(enumerate(query_loader))):
        pbar.set_description(f"Processing query batch {i}")
        
        features = model(q.to(device)) #normalized
        all_q_features = torch.cat([all_q_features, features], dim=0)
        all_qids = torch.cat([all_qids, q_id.to(torch.int)])
        all_qcamids = torch.cat([all_qcamids, q_camid.to(torch.int)])
        
    torch.cuda.empty_cache()
    
    all_g_features = torch.empty((0,EMBEDDING_DIM), dtype=torch.float, device=device)
    all_gids = torch.empty((0,), dtype=torch.int)
    all_gcamids = torch.empty((0,), dtype=torch.int)
    for g, g_id, g_camid in (pbar := tqdm(gallery_loader)):
        pbar.set_description(f"Processing gallery batch {i}")
        #g.to(device)
        features = model(g.to(device)) #normalized
        all_g_features = torch.cat([all_g_features, features], dim=0)
        all_gids = torch.cat([all_gids, g_id.to(torch.int)])
        all_gcamids = torch.cat([all_gcamids, g_camid.to(torch.int)])
        
        
    result = {"gallery_f" : all_g_features.cpu().numpy(),
              "gallery_ids" : all_gids.numpy(),
              "gallery_camids" : all_gcamids.numpy(),
              "query_f" : all_q_features.cpu().numpy(),
              "query_ids" : all_qids.numpy(),
              "query_camids" : all_qcamids.nump()
    }
    log(f"saving matrix...")
    savemat(MATRIX_FILEPATH, result)
    log(f"saved matrix.")
        
        


if __name__ == "__main__":
    torch.cuda.empty_cache()
    model = Net(num_classes=NUM_CLASSES, normalize=True)
    model.to(DEVICE)
    state = torch.load(WEIGHTS_PATH, map_location="cuda")

    model.load_state_dict(state["model_state_dict"])
    
    
    
    log(f"Loaded model.")
    
    test_transform = T.Compose([T.ToTensor(), T.Resize(size=(256,128))])
    query_set, gallery_set = get_query_test_loaders(test_transform)
    
    
    query_loader = DataLoader(query_set, batch_size=64, shuffle=False, drop_last=False, pin_memory=False)
    gallery_loader = DataLoader(gallery_set, batch_size=64, shuffle=False, drop_last=False, pin_memory=False)
    
    log(f"Loaded query and gallery dataloaders.")
    
    save_feature_matrix(model, query_loader, gallery_loader)