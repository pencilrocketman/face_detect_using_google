import numpy as np
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import glob
import os
import random
 
in_dir = "/home/masa/pytmp3/origin_image/*"
out_dir = "/home/masa/pytmp3/face_image"
in_jpg=glob.glob(in_dir)
in_fileName=os.listdir("/home/masa/pytmp3/origin_image/")
print(len(in_jpg))
for num in range(len(in_jpg)):
    image=cv2.imread(str(in_jpg[num]))
    if image is None:
        continue
    
    image_gs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier("/home/masa/pytmp3/opencv/opencv-3.3.0/data/haarcascades/haarcascade_frontalface_alt.xml")
    face_list=cascade.detectMultiScale(image_gs, scaleFactor=1.1, minNeighbors=2,minSize=(64,64))
    if len(face_list) > 0:
        for rect in face_list:
            x,y,width,height=rect
            image = image[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]]
            if image.shape[0]<64:
                continue
            image = cv2.resize(image,(64,64))
    else:
        print("no face")
        continue
    print(image.shape)
    #保存
    fileName=os.path.join(out_dir,str(in_fileName[num]))
    cv2.imwrite(str(fileName),image)

in_dir = "/home/masa/pytmp3/face_image/*"
in_jpg=glob.glob(in_dir)
img_file_name_list=os.listdir("/home/masa/pytmp3/face_image/")

random.shuffle(in_jpg)
import shutil
for i in range(len(in_jpg)//5):
    shutil.move(str(in_jpg[i]), "/home/masa/pytmp3/test_image")
