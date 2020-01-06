import os
from PIL import Image
import numpy as np
import cv2
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "images")

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

currentID = 0
label_ids = {}

y_label = []
x_train = []

for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root, file)
            label = os.path.basename(root).replace("_", "-").lower()
            # print(label, path)
            if label not in label_ids:
                label_ids[label] = currentID
                currentID += 1

            id_ = label_ids[label]
            # print(label_ids)
            # x_train.append(path)        # verify this array, and turn into numpy array, Gray
            # y_label.append(label)       # use numbers instead
            # covert images into numbers -> for each pixel
            pil_image = Image.open(path).convert("L")  # grayscale
            size = (550, 550)
            final_image = pil_image.resize(size, Image.ANTIALIAS)
            image_arr = np.array(final_image, "uint8")
            # print(image_arr)
            faces = face_cascade.detectMultiScale(image_arr, scaleFactor=1.5, minNeighbors=5)

            for (x, y, w, h) in faces:
                roi = image_arr[y:y + h, x:x + h]
                x_train.append(roi)
                y_label.append(id_)

with open("labels.pkl", "wb") as f:
    pickle.dump(label_ids, f)

recognizer.train(x_train, np.array(y_label))
recognizer.save("trainer.yml")
