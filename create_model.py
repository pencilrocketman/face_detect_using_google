import random
from keras.utils.np_utils import to_categorical
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

img_file_name_list=os.listdir("./face_scratch_image/")
print(len(img_file_name_list))

for i in range(len(img_file_name_list)):
    n=os.path.join("./face_scratch_image",img_file_name_list[i])
    img = cv2.imread(n)
    if isinstance(img,type(None)) == True:
        img_file_name_list.pop(i)
        continue
print(len(img_file_name_list))

X_train=[] # image
y_train=[] # label 
for j in range(0,len(img_file_name_list)-1):
    n=os.path.join("./face_scratch_image/",img_file_name_list[j])
    img = cv2.imread(n)
    b,g,r = cv2.split(img)
    img = cv2.merge([r,g,b])
    X_train.append(img)
    n=img_file_name_list[j]
    y_train=np.append(y_train,int(n[0:2])).reshape(j+1,1)

X_train=np.array(X_train)

img_file_name_list=os.listdir("./test_image/")
print(len(img_file_name_list))

for i in range(len(img_file_name_list)):
    n=os.path.join("./test_image",img_file_name_list[i])
    img = cv2.imread(n)
    if isinstance(img,type(None)) == True:
        img_file_name_list.pop(i)
        continue
print(len(img_file_name_list))

X_test=[]
y_test=[]

for j in range(0,len(img_file_name_list)):
    n=os.path.join("./test_image",img_file_name_list[j])
    img = cv2.imread(n)
    b,g,r = cv2.split(img)
    img = cv2.merge([r,g,b])
    X_test.append(img)
    n=img_file_name_list[j]
    y_test=np.append(y_test,int(n[0:2])).reshape(j+1,1)
    
X_test=np.array(X_test)

from keras.layers import Activation, Conv2D, Dense, Flatten, MaxPooling2D
from keras.models import Sequential, load_model

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = Sequential()
model.add(Conv2D(input_shape=(64, 64, 3), filters=32,kernel_size=(2, 2), strides=(1, 1), padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=32, kernel_size=(2, 2), strides=(1, 1), padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=32, kernel_size=(2, 2), strides=(1, 1), padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(256))
model.add(Activation("sigmoid"))
model.add(Dense(128))
model.add(Activation('sigmoid'))
model.add(Dense(5))
model.add(Activation('softmax'))

model.compile(optimizer='sgd', loss='categorical_crossentropy',metrics=['accuracy'])

# model.fit(X_train, y_train, batch_size=32, epochs=50)

history = model.fit(X_train, y_train, batch_size=32, epochs=80, verbose=1, validation_data=(X_test, y_test))

score = model.evaluate(X_test, y_test, batch_size=32, verbose=0)
print('validation loss:{0[0]}\nvalidation accuracy:{0[1]}'.format(score))

plt.plot(history.history["acc"], label="acc", ls="-", marker="o")
plt.plot(history.history["val_acc"], label="val_acc", ls="-", marker="x")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(loc="best")
plt.show()

model.save("my_model.h5")

import numpy as np
import matplotlib.pyplot as plt

def detect_face(image):
    print(image.shape)
    image_gs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier("/home/masa/pytmp3/opencv/opencv-3.3.0/data/haarcascades/haarcascade_frontalface_alt.xml")
    face_list=cascade.detectMultiScale(image_gs, scaleFactor=1.1, minNeighbors=2,minSize=(64,64))
    # when detect face
    if len(face_list) > 0:
        for rect in face_list:
            x,y,width,height=rect
            cv2.rectangle(image, tuple(rect[0:2]), tuple(rect[0:2]+rect[2:4]), (255, 0, 0), thickness=3)
            img = image[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]]
            if image.shape[0]<64:
                print("too small")
                continue
            img = cv2.resize(image,(64,64))
            img=np.expand_dims(img,axis=0)
            name = detect_who(img)
            cv2.putText(image,name,(x,y+height+20),cv2.FONT_HERSHEY_DUPLEX,1,(255,0,0),2)
    # when not detect face
    else:
        print("no face")
    return image
    
def detect_who(img):
    name=""
    print(model.predict(img))
    nameNumLabel=np.argmax(model.predict(img))
    if nameNumLabel== 0: 
        name="Techi"
    elif nameNumLabel==1:
        name="Peh"
    elif nameNumLabel==2:
        name="Neru"
    elif nameNumLabel==3:
        name="Monta"
    elif nameNumLabel==4:
        name="Yukkah"
    return name

image=cv2.imread("./origin_image/01.0.jpg")
if image is None:
    print("Not open:")
b,g,r = cv2.split(image)
image = cv2.merge([r,g,b])
whoImage=detect_face(image)

plt.imshow(whoImage)
plt.show()
