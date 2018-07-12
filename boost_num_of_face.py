import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob

def scratch_image(img, flip=True, thr=True, filt=True):
    methods = [flip, thr, filt]
    filter1 = np.ones((3, 3))
    images = [img]
    scratch = np.array([
        lambda x: cv2.flip(x, 1),
        lambda x: cv2.threshold(x, 100, 255, cv2.THRESH_TOZERO)[1],
        lambda x: cv2.GaussianBlur(x, (5, 5), 0),
    ])
    doubling_images = lambda f, imag: np.r_[imag, [f(i) for i in imag]]

    for func in scratch[methods]:
        images = doubling_images(func, images)
    return images
    
in_dir = "/home/masa/pytmp3/face_image/*"
in_jpg=glob.glob(in_dir)
img_file_name_list=os.listdir("/home/masa/pytmp3/face_image/")
for i in range(len(in_jpg)):
    print(str(in_jpg[i]))
    img = cv2.imread(str(in_jpg[i]))
    scratch_face_images = scratch_image(img)
    for num, im in enumerate(scratch_face_images):
        fn, ext = os.path.splitext(img_file_name_list[i])
        file_name=os.path.join("/home/masa/pytmp3/face_scratch_image",str(fn+"."+str(num)+".jpg"))
        cv2.imwrite(str(file_name) ,im)
