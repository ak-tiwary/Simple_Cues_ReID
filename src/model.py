import torch
import torch.nn as nn
import torchvision


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x
    
    
def get_backbone():
    """Can be modified to change backbone."""
    resnet50 = torchvision.models.resnet50(weights="DEFAULT")
    
    for i, layer in enumerate(resnet50.modules()):
        if isinstance(layer, nn.BatchNorm2d):
            #use batch stats during inference
            layer.track_running_stats = False 
            
    #dummy variable to transport info
    resnet50._infeatures_temp = resnet50.fc.in_features
    #discard fc layer
    resnet50.fc = Identity()
    
    return resnet50
    
    
class Net(nn.Module):
    """The full Siamese network with BNNeck."""
    def __init__(self, num_classes, backbone=get_backbone()):
        """num_classes is the number of identities being classified."""
        #gives features, used to calculate triplet and center loss
        self.backbone = backbone 
        
        in_features = backbone._infeatures_temp
        
        #output of this is used during inference
        self.batch_norm = nn.BatchNorm2d(num_features=in_features,
                                         track_running_stats=False)
        
        self.fc = nn.Linear(in_features=in_features, 
                            out_features=num_classes,
                            bias=False)
        
    def forward(self, x):
        """During test time returns features and output of fc layers.
        During inference time returns output after batch normalization.
        
        Potential minor issue: When in validation mode should the model
        act as if in inference mode and return batch normalized outputs?"""
        
        features = self.backbone(x)
        
        if self.training:
            class_probs = self.fc(self.batch_norm(features))
            return class_probs, features 
        else: #inference mode
            return self.batch_norm(features)

