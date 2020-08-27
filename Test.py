from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np

#giving path to cascade classifier i.e xml file,CascadeClassifier will create a CascadeClssifier Object 
face_classifier =cv2.CascadeClassifier(r'C:\Users\anand\Desktop\Mini Project\emotion_detection-master\haarcascade_frontalface_default.xml')

#loading the model which will be used for prediction
classifier =load_model(r'C:\Users\anand\Desktop\Mini Project\emotion_detection-master\Emotion_little_vgg.h5')

class_labels = ['Angry','Happy','Neutral','Sad','Surprise']

cap = cv2.VideoCapture(0)

while True:
    # Grab a single frame of video
    ret, frame = cap.read()
    labels = []
    #converting the image(here frame to GrayScale)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #method to search for face rectangle co-ordinates-->detectMultiScale, 
    #1.3 scale factor it decreases the shape value until the face is found.
    #5->minneighbours specifying how many neighbors each candidate rectangle should have to retain it.
    faces = face_classifier.detectMultiScale(gray,1.3,5)

    #cv2.rectangle(start pt,end pt, color of the border, thickness of the border)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h,x:x+w] #whatever inside the frame(image)
        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA) #shrinking the image, hence using INTER_AREA interpolation.
    # rect,face,image = face_detector(frame)


        if np.sum([roi_gray])!=0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)

        # make a prediction on the ROI, then lookup the class
        # Function np.argmax() returns the index of the maximum value, not the value.
            preds = classifier.predict(roi)[0]
            label=class_labels[preds.argmax()]
            label_position = (x,y)
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
        else:
            cv2.putText(frame,'No Face Found',(20,60),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
    cv2.imshow('Mini Project Emotion Detection',frame)
    #waitkey(1) will take a new frame every 1 millisecond
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


























