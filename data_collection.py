import os
import time 
import uuid
import cv2


PATH = "data_raw/complete/images"; 

cam = cv2.VideoCapture(0); 
number_images = 0; 
while True:
    user_input = input("Press Enter to take a picture!");
    if user_input == 'q': break; 
    print("Taking Picture.."); 
    number_images += 1; 

    ret, frame = cam.read(); 
    image_name = os.path.join(PATH, f"{str(uuid.uuid1())}.jpg"); 
    cv2.imwrite(image_name, frame); 
    cv2.imshow('frame', frame); 

cam.release(); 
cv2.destroyAllWindows(); 
print(f"Finished taking {number_images} Images!"); 



