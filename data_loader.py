import torch 
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import cv2
import json 
import os
from matplotlib import pyplot as plt
import numpy as np

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

for images, labels in train_loader: 
    print(type(images)); 
    print("Images Shape:", images.shape);  
    print(type(labels)); 
    print(labels); 

    fig, ax = plt.subplots(ncols=4);
    for idx in range(4):
        sample_image = images[idx].permute(1, 2, 0).numpy().copy(); 
#        print(sample_image.shape); 
        sample_coords = [float(labels[1][0][idx]), float(labels[1][1][idx]), float(labels[1][2][idx]), float(labels[1][3][idx])]; 
        cv2.rectangle(sample_image, tuple(np.multiply(sample_coords[:2], [120, 120]).astype(int)),
                                    tuple(np.multiply(sample_coords[2:], [120, 120]).astype(int)),
                                    (255, 0, 0), 2); 
        ax[idx].imshow(sample_image); 
    plt.show(); 
    break; 


print("Finished!"); 
