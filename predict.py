import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image
import cv2 
import json
import os
from matplotlib import pyplot as plt
import numpy as np
import torch.nn.functional as F

class CustomDataSet(Dataset): 
    def __init__(self, directory, transform=None, target_transform=None):
        self.directory = directory; 
        self.image_names = os.listdir(f"{directory}/images"); 
        self.transform = transform;
        self.target_transform = target_transform; 

    def __len__(self): 
        return len(os.listdir(f"{self.directory}/images")); 

    def __getitem__(self, idx): 
        image = read_image(f"{self.directory}/images/{self.image_names[idx]}");
        with open(f"{self.directory}/labels/{self.image_names[idx].split('.')[0]}.{self.image_names[idx].split('.')[1]}.json", 'r') as f: 
            label = json.load(f);  
        label = [label['class']], label['bbox']; 
        if self.transform: 
            image = self.transform(image); 
        if self.target_transform: 
            label = self.target_transform(label);
        return image, label; 
    
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


transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((120, 120)),
    torchvision.transforms.Lambda(lambda x: x/255) 
]); 
#test_images = CustomDataSet("data_aug_720x720/test", transform=transforms); 
#test_loader = DataLoader(test_images, batch_size=8, shuffle=True); 

net = torch.load("models/face_detector_12.pth"); 
net.eval(); 



with torch.no_grad(): 
    ##############################################################
    ######### Predict Sampled from the Test Set ##################
#    for X, _ in test_loader: 
#        pred_class, pred_bbox = net(X);  
#        print(pred_class); 
#        print(pred_bbox); 
#        print(pred_class[0]); 
#        print(pred_bbox[0,:]); 
##        quit(); 
#        fig, ax = plt.subplots(ncols=4); 
#        for idx in range(4): 
#            sample_image = X[idx].permute(1, 2, 0).numpy().copy(); 
#            sample_class = pred_class[idx].item(); 
#            sample_bbox = pred_bbox[idx,:].numpy().copy(); 
#            cv2.rectangle(sample_image, tuple(np.multiply(sample_bbox[:2], [120, 120]).astype(int)),
#                                    tuple(np.multiply(sample_bbox[2:], [120, 120]).astype(int)),
#                                    (255, 0, 0), 2); 
#            ax[idx].imshow(sample_image); 
#        plt.show(); 
#        break; 
    ##############################################################
    ######### Live Prediction ####################################
    cam = cv2.VideoCapture(0); 
    while cam.isOpened(): 
        _, frame = cam.read(); 
        frame = frame[:, 280:1000, :]; 
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB); 

        image = transforms(torch.tensor(frame).permute(2, 0, 1)); 
#        print(image.shape); 
        
        pred_class, pred_bbox = net(image); 

        sample_class = pred_class[0].item(); 
        sample_bbox = pred_bbox[0,:].numpy().copy(); 
        
        if sample_class > 0.5: 
            #Main Rectangle
            cv2.rectangle(frame, tuple(np.multiply(sample_bbox[:2], [720, 720]).astype(int)),
                                        tuple(np.multiply(sample_bbox[2:], [720, 720]).astype(int)),
                                        (255, 0, 0), 2); 
            cv2.putText(frame, 'chris', tuple(np.add(np.multiply(sample_bbox[:2], [720, 720]).astype(int),
                [0, -5])),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA); 
        cv2.imshow("FaceTracker", frame); 

    ##############################################################

        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break; 
cam.release(); 
cv2.destroyAllWindows(); 

































