import cv2
import numpy as np
from PIL import Image
import os

#DETECTING FACES
def draw_boundary(img, classifier, scalefactor, minNeighbor, color, txt, clf):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    features = classifier.detectMultiScale(gray_img, scalefactor, minNeighbor)
    coords = [] 
    
    for(x,y,w,h) in features:
        cv2.rectangle(img, (x,y), (x+w,y+h), color, 2)
        id,pred = clf.predict(gray_img[y:y+h, x:x+w])
        
        confidence = int(100 * (1-pred/300))
        if confidence>77:
            if id == 1:
                cv2.putText(img, "person_name", (x,y), cv2.FONT_HERSHEY_COMPLEX, 0.8, color, 1, cv2.LINE_AA)
            
            # multiple if for multiple authorized person 
        else:
            cv2.putText(img, "UNKNOWN", (x,y), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0,0,255), 1, cv2.LINE_AA)
            
        coords = [x,y,w,h]
        
    return coords

def recognize(img, clf, faceCascade):
    coords = draw_boundary(img, faceCascade, 1.2, 8, (255,255,255), "Face", clf)
    return img

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
clf = cv2.face.LBPHFaceRecognizer_create()
clf.read("classifier.xml")

videoCapture = cv2.VideoCapture(1) # 1 for external camera, 0 for inbuild camera (0-back cam, 1-front cam)

while True:
    ret,img = videoCapture.read()
    img = recognize(img, clf, faceCascade)
    cv2.imshow("face detection", img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
videoCapture.release()
cv2.destroyAllWindows()