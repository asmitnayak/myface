import numpy as np
import cv2
import pickle
import os
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "images")

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

labels = {}
with open("labels.pkl", "rb") as f:
    og_labels = pickle.load(f)
    labels = {v:k for k, v in og_labels.items()}

cap = cv2.VideoCapture(0)

counter = 0

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=6)

    for (x, y, w, h) in faces:
        counter += 1
        print(x, y, w, h)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y + h, x:x + w]

        # recognize -> deep learn model -> scikit learn, etc
        id_, conf = recognizer.predict(roi_gray)
        if 50 <= conf:
            print(id_)
            print(labels[id_])
            font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
            name = labels[id_]
            color = (255, 255, 255)
            stroke = 1
            cv2.putText(frame, name, (x, y), font, 1, color, stroke, cv2.LINE_AA)
        print(conf)
        img_item = "my-image.png"
        cv2.imwrite(img_item, roi_gray)

        color = (255, 0, 0)   # BGR 0-255
        stroke = 2
        end_cord_x = x + w;
        end_cord_y = y + h;
        cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)

        if counter == 100:
            prmt = "IS THE DETECTED FACE THAT OF " + labels[id_] + "? (Y/N): "
            user_inp = input(prmt)
            if user_inp == 'Y' or user_inp == 'y':
                img_label_dir = os.path.join(image_dir, labels[id_])
                new_name = str(datetime.now()).replace(" ", "_").replace(":", "_")
                completeName = "images/" + str(labels[id_]) + "/" + new_name + ".png"
                print(completeName)
                cv2.imwrite(str(completeName), roi_color)
                exec(open("faces_train.py").read())
                with open("labels.pkl", "rb") as f:
                    og_labels = pickle.load(f)
                    labels = {v: k for k, v in og_labels.items()}

            counter = 0
        print(counter)

    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything is done
cap.release();
cv2.destroyAllWindows()
