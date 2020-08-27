from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image

#from keras.models import load_model

from tensorflow.keras.models import load_model
import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#loading the model which will be used for prediction
classifier =load_model(r'Emotion_little_vgg.h5')

class_labels = ['Angry','Happy','Neutral','Sad','Surprise']

class VideoCamera(object):
    def __init__(self):  
        self.video=cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, image = self.video.read()
        labels = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        #for (x,y,w,h) in faces:
        #    cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
        #    roi_gray = gray[y:y+h, x:x+w]
        #    roi_color = image[y:y+h, x:x+w]
        #for (x, y, w, h) in faces:
        #   gray_face = cv2.resize((gray[y:y + h, x:x + w]), (110, 110))
        #   eyes = eye_cascade.detectMultiScale(gray_face)
        #   for (ex, ey, ew, eh) in eyes:

        #        draw_box(gray, x, y, w, h)

        for (x,y,w,h) in faces:
            cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h,x:x+w]
            roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)

            
            if np.sum([roi_gray])!=0:
                roi = roi_gray.astype('float')/255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi,axis=0)

        # make a prediction on the ROI, then lookup the class

                preds = classifier.predict(roi)[0]
                label=class_labels[preds.argmax()]
                label_position = (x,y)
                cv2.putText(image,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)

        ret, jpeg = cv2.imencode('.jpg', image)

        return jpeg.tobytes()