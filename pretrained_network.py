import torch 
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import json 
import os
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

transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((120, 120)),
    torchvision.transforms.Lambda(lambda x: x/255) 
]); 

train_images = CustomDataSet("data_aug_720x720/train", transform=transforms); 

train_loader = DataLoader(train_images, batch_size=8, shuffle=True); 

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


def format_bbox_labels(labels): 
    catted = torch.stack(labels).to(torch.float32); 
    return catted.transpose(0, 1); 

def localization_loss(yhat, ytrue): 
    delta_coord = torch.sum(torch.square(ytrue[:,:2] - yhat[:,:2])); 

    h_true = ytrue[:,3] - ytrue[:,1]; 
    w_true = ytrue[:,2] - ytrue[:,0];

    h_pred = yhat[:,3] - yhat[:,1]; 
    w_pred = yhat[:,2] - yhat[:,0];
   
    delta_size = torch.sum(torch.square(w_true - w_pred) + torch.square(h_true - h_pred)); 
    return delta_coord + delta_size; 


net = VGG16DualHead(); 
optimizer = optim.Adam(net.parameters(), lr=0.0001); 
scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer, lambda epoch: 0.75); 

criterion_class = torch.nn.BCELoss(); 
criterion_bbox = localization_loss; 

EPOCHS = 30; 

for epoch in range(EPOCHS): 
    for i, (X, y) in enumerate(train_loader): 
        y_class = y[0][0].reshape(-1, 1).to(torch.float32); 
        y_bbox = format_bbox_labels(y[1]); 

        pred_class, pred_bbox = net(X); 
         
        loss_class = criterion_class(pred_class, y_class);
        loss_bbox = criterion_bbox(pred_bbox, y_bbox);  
        batch_loss = loss_class + 0.5*loss_bbox; 

        net.zero_grad(); 
        batch_loss.backward(); 
        optimizer.step(); 
        if i % 10 == 0: 
            print(f"Epoch #{epoch} Batch #{i} Loss: {batch_loss}"); 
    scheduler.step(); # lr *= 0.75;  
    torch.save(net, f"models/face_detector_{epoch}.pth"); 


























