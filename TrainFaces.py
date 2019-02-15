import os;
from PIL import Image;
import cv2;
import numpy as np;
import pickle;

face_cascades=cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml');
recognizer=cv2.face.LBPHFaceRecognizer_create();
BASE_PATH=os.path.dirname(os.path.abspath(__file__));
img_dir=os.path.join(BASE_PATH,'Images');
current_id=0;
label_ids={};
y_labels=[];
x_train=[];
for root,dirs,files in os.walk(img_dir):
    for file in files:
        if(file.endswith('png') or file.endswith('jpg')):
            path=os.path.join(root,file);
            label=os.path.basename(os.path.dirname(path).replace(' ','-').lower());
            #print(label,path);
            if(label in label_ids):
                pass;
            else:
                label_ids[label]=current_id;
                current_id+=1;
            id_=label_ids[label];
            pil_image=Image.open(path).convert('L'); #gray scale
            # size=(700,700);
            #final_img=pil_image.resize(size,Image.ANTIALIAS);

            image_array=np.array(pil_image,"uint8");
           # print(image_array);

            faces=face_cascades.detectMultiScale(image_array,scaleFactor=1.5,minNeighbors=2);

            for(x,y,w,h) in faces:
                roi=image_array[y:y+h,x:x+h];
                x_train.append(roi);
                y_labels.append(id_);
                print(y_labels);

            #print(x_train);

print(label_ids);
with open('labels.pickle','wb') as f:
    pickle.dump(label_ids,f);

recognizer.train(x_train,np.array(y_labels));
recognizer.save('trainner.yml');