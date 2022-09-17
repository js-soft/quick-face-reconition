import numpy as np
from matplotlib import pyplot as plt
import json 
import os
import cv2
import albumentations as alb

cropwidth, cropheight = (720, 720); 

augmentor = alb.Compose([
    alb.RandomCrop(width=cropwidth, height=cropheight),
    alb.HorizontalFlip(p=0.5),
    alb.RandomBrightnessContrast(p=0.2),
    alb.RandomGamma(p=0.2),
    alb.RGBShift(p=0.2),
    alb.VerticalFlip(p=0.5)
], bbox_params=alb.BboxParams(format='albumentations', label_fields=['class_labels']));

#for partition in ["train", "test", "val"]: 
if True: 
#    for image_name in os.listdir(f"data_raw/{partition}/images"): 
#        img = cv2.imread(f"data_raw/{partition}/images/{image_name}");
    for image_name in os.listdir(f"data_example/raw/images"): 
        img = cv2.imread(f"data_example/raw/images/{image_name}");
        coords = [0, 0, 0.00001, 0.00001]; 
#        label_path = f"data_raw/{partition}/labels/{image_name.split('.')[0]}.json"; 
        label_path = f"data_example/raw/labels/{image_name.split('.')[0]}.json"; 
        if os.path.exists(label_path): 
            with open(label_path, 'r') as f: 
                label = json.load(f); 
            
            coords[0] = label["shapes"][0]["points"][0][0]; 
            coords[1] = label["shapes"][0]["points"][0][1]; 
            coords[2] = label["shapes"][0]["points"][1][0]; 
            coords[3] = label["shapes"][0]["points"][1][1]; 
            coords = list(np.divide(coords, [1280, 780, 1280, 780])); 
        try: 
            for x in range(60): 
                augmented = augmentor(image=img, bboxes=[coords], class_labels=['chris']); 
#                cv2.imwrite(f"data_aug/{partition}/images/{image_name.split('.')[0]}.{x}.jpg", augmented["image"]); 
                cv2.imwrite(f"data_example/augmented/images/{image_name.split('.')[0]}.{x}.jpg", augmented["image"]); 
                
                annotation = {}; 
                annotation['image'] = image_name; 

                if os.path.exists(label_path): 
                    if len(augmented['bboxes']) == 0: 
                        annotation['bbox'] = [0, 0, 0, 0]; 
                        annotation['class'] = 0; 
                    else: 
                        annotation['bbox'] = augmented['bboxes'][0]; 
                        annotation['class'] = 1; 
                else: 
                    annotation['bbox'] = [0, 0, 0, 0]; 
                    annotation['class'] = 0; 

#                with open(f"data_aug/{partition}/labels/{image_name.split('.')[0]}.{x}.json", 'w') as f: 
                with open(f"data_example/augmented/labels/{image_name.split('.')[0]}.{x}.json", 'w') as f: 
                    json.dump(annotation, f); 
                
        except Exception as e: 
            print(e); 

print("Finished!"); 



