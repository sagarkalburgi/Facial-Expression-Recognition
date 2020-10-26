import cv2
from tensorflow.keras.models import model_from_json
import numpy as np

facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = model_from_json(open('model.json', 'r').read())
model.load_weights('model_weights.h5')
font = cv2.FONT_HERSHEY_COMPLEX
emotions = ('Angry', 'Disgust','Fear', 'Happy', 'Sad', 'Surprise', 'Neutral')

cap = cv2.VideoCapture(0)

# returns camera frames along with bounding boxes and predictions
while cap.isOpened():
    _, fr = cap.read()
    gray_fr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    faces = facec.detectMultiScale(gray_fr, 1.1, 4)

    for (x, y, w, h) in faces:
        fc = gray_fr[y:y+h, x:x+w]

        roi = cv2.resize(fc, (48, 48))
        pred = model.predict(roi[np.newaxis, :, :, np.newaxis])
        max_index = np.argmax(pred[0])
        emotions = ('Angry', 'Disgust','Fear', 'Happy', 'Sad', 'Surprise', 'Neutral')
        emotions = emotions[max_index]
        emotions = emotions + ': ' + str(round(pred[0][max_index]*100,2))

        cv2.putText(fr, emotions, (x, y), font, 2, (255, 0, 0), 3)
        cv2.rectangle(fr,(x,y),(x+w,y+h),(0,0,255),5)

    cv2.imshow('img', fr)
    if cv2.waitKey(1) == ord('q'): #press q to quit
        break  

cap.release()
cv2.destroyAllWindows()
