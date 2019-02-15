import numpy as np;
import cv2;
import pickle;

face_cascades=cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml');
recognizer=cv2.face.LBPHFaceRecognizer_create();
recognizer.read('trainner.yml');

with open('labels.pickle','rb') as f:
    pre_labels=pickle.load(f);
    labels={v:k for k,v in pre_labels.items()};

cap=cv2.VideoCapture(0);
i=0;
while(True):
    ret,frame=cap.read();
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY);
    faces=face_cascades.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=20);
    for (x,y,w,h) in faces:
        #print(x,y,w,h);
        roi_gray=gray[y:y+h,x:x+w];
        roi_color=frame[y:y+h,x:x+w];

        #recognizer
        id_,conf=recognizer.predict(roi_gray);
        if(conf>=30 and conf<=80):
            print(id_);
            print(labels[id_]);
            font=cv2.FONT_HERSHEY_SIMPLEX;
            name=labels[id_];
            color=(255,255,255);
            stroke=2;
            cv2.putText(frame,name,(x,y),font,1,color,stroke,cv2.LINE_AA);
        print(conf);
        img_item='8.png';
        cv2.imwrite(img_item,roi_color);

        color=(255,0,0); #BGR
        stroke=2;
        end_cord_x=x+w;
        end_cord_y=y+h;
        cv2.rectangle(frame,(x,y),(end_cord_x,end_cord_y),color,stroke);


    cv2.imshow('Video-1',frame);
    if(cv2.waitKey(20) & 0xff==ord('q')):
        if (i < 100):
            cv2.imwrite('cap' + str(i) + '.jpg', gray);
            i += 1;
cap.release();
cv2.destroyAllWindows();