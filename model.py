import torch 
import torchvision
import torch.nn as nn
import torch.nn.functional as F


vgg16 = torchvision.models.vgg16(pretrained=True); 

class VGG16DualHead(nn.Module): 
    def __init__(self): 
        super().__init__(); 
        self.vgg16_base = vgg16.features; 
        
        self.classification1 = nn.Linear(512, 2048); 
        self.classification2 = nn.Linear(2048, 1); 
        
        self.regression1 = nn.Linear(512, 2048); 
        self.regression2 = nn.Linear(2048, 4); 
        
    def forward(self, X):
        X = self.vgg16_base(X); 
        X = F.max_pool2d(X, kernel_size=(3, 3)).reshape(-1, 512); 

        X_class = F.relu(self.classification1(X)); 
        X_class = torch.sigmoid(self.classification2(X_class)); 

        X_reg = F.relu(self.regression1(X)); 
        X_reg = torch.sigmoid(self.regression2(X_reg)); 

        return X_class, X_reg; 





