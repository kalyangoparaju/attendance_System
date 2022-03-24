import os
import cv2
import face_recognition
import numpy as np

path='images'
images=[]
names=[]
lis=os.listdir(path)


for i in lis:
    im=cv2.imread(f'{path}/{i}')
    images.append(im)
    names.append(os.path.splitext(i)[0])


f=open('encoding.txt','w')
def findencoding(images):
    encodelist=[]
    for img in images:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        encode=face_recognition.face_encodings(img)[0]
        encodelist.append(encode)
    return encodelist

encodelistknown=findencoding(images)
np.savetxt('encoding.txt', encodelistknown, fmt='%f')
