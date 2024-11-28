import numpy as np
from PIL import Image
import os
import cv2

# TRAINING CLASSIFIER
def train_classifier(data_dir):
    path = [os.path.join(data_dir,image) for image in os.listdir(data_dir)]
    faces = []
    ids = []
    
    for image in path:
        img = Image.open(image).convert('L')
        imageNp = np.array(img, 'unit8')
        id = int(os.path.split(image)[1].split(".")[1]) #image = C:\Users\ravir\Desktop\bytelocker\data\user.1.1
        
        faces.append(imageNp)
        ids.append(id)
    
    ids = np.array(ids)
    
    #classifier
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces,ids)
    clf.write("classifier.xml")
    
    
train_classifier("data")