import numpy as np;
import cv2;
import pickle;

face_cascades=cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml');

cap=cv2.VideoCapture(0);

while(True):
    ret,frame=cap.read();
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY);
    faces=face_cascades.detectMultiScale(gray,scaleFactor=1.5,minNeighbors=2);
    for (x,y,w,h) in faces:
        #print(x,y,w,h);
        roi_gray=gray[y:y+h,x:x+w];
        roi_color=frame[y:y+h,x:x+w];
        img_item='8.png';
        cv2.imwrite(img_item,roi_color);

        color=(255,0,0); #BGR
        stroke=2;
        end_cord_x=x+w;
        end_cord_y=y+h;
        cv2.rectangle(frame,(x,y),(end_cord_x,end_cord_y),color,stroke);


    cv2.imshow('vid-1',frame);
    if(cv2.waitKey(20) & 0xff==ord('q')):
        break;

cap.release();
cv2.destroyAllWindows();



