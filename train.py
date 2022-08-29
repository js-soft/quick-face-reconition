#Imports from Modules
import torch 
import torch.optim as optim
from torch.utils.data import DataLoader
#Imports from local Files
from CustomDataSet import CustomDataSet, transforms
from model import VGG16DualHead

train_images = CustomDataSet("data_aug_720x720/train", transform=transforms); 
train_loader = DataLoader(train_images, batch_size=8, shuffle=True); 

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

EPOCHS = 30;    #höhö 

#Training loop ist nicht sehr schön und gibt einem sehr wenige stats aber provisorisch tut er
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


























