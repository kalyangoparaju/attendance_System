import cv2
from cv2 import waitKey
import numpy as np
import face_recognition
import os
import pandas as pd
import json 
import time



path='images'
names=[]
lis=os.listdir(path)


for i in lis:
    names.append(os.path.splitext(i)[0])


dct1={'name':{},'attendance':{}}  #!  comment it
cnt=0
for i in names:
    dct={}
    dct[cnt]=i
    dct1['name'].update(dct)
    dct[cnt]='unmarked'
    dct1['attendance'].update(dct)
    cnt+=1
with open('status.json','w') as json_file:
    json.dump(dct1,json_file)


encodelistknown = np.loadtxt('encoding.txt', dtype=np.float64)

print("encoding complete")

encodetime=time.time()


with open("status.json","r") as f:
    data=json.load(f)

def markattendance(name):
        for nm,mar in zip(data['name'],data['attendance']):
            if data['name'][nm]==name and data['attendance'][mar]=='unmarked':
                data['attendance'][mar]='present'


grouppath='test'
grouplis=os.listdir(grouppath)


for one in grouplis:

    frame=cv2.imread(f'{grouppath}/{one}')
    small=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    faces=face_recognition.face_locations(small)
    currentencode=face_recognition.face_encodings(small,faces)

    for encodeface,faceloc in zip(currentencode,faces):
        matches=face_recognition.compare_faces(encodelistknown,encodeface)
        facedis=face_recognition.face_distance(encodelistknown,encodeface)
        matchdis=np.argmin(facedis)

        if matches[matchdis] :
            name=names[matchdis]
            markattendance(name)


with open("status.json",'w') as f:
    json.dump(data,f)



df = pd.read_json (r'status.json')
df.to_csv (r'attendence.csv', index = None)