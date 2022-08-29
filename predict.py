#Imports from Modules
import torch
from torch.utils.data import DataLoader
import cv2 
from matplotlib import pyplot as plt
#Imports from local Files
import numpy as np
from CustomDataSet import CustomDataSet, transforms
from model import VGG16DualHead

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
        
        pred_class, pred_bbox = net(image); 

        sample_class = pred_class[0].item(); 
        sample_bbox = pred_bbox[0,:].numpy().copy(); 
        
        if sample_class > 0.5: 
            #Main Rectangle
            cv2.rectangle(frame, tuple(np.multiply(sample_bbox[:2], [720, 720]).astype(int)),
                                        tuple(np.multiply(sample_bbox[2:], [720, 720]).astype(int)),
                                        (255, 0, 0), 2); 
            #Label Text
            cv2.putText(frame, 'chris', tuple(np.add(np.multiply(sample_bbox[:2], [720, 720]).astype(int),
                [0, -5])),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA); 
        cv2.imshow("FaceTracker", frame); 
    ##############################################################
    #Idk why this is here but without it the image isn't displayed
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break; 
cam.release(); 
cv2.destroyAllWindows(); 

































