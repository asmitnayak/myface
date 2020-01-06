from imutils import paths
import face_recognition
import pickle
import cv2
import os
import sys
import time

# CNN: Convolution Neural Network       -> slow (without GPU)
# HoG: Histogram of Oriented Gradient   -> fast (use in Raspberry Pi)
m = "hog"


def animated_loading(mes='Encoding in process...'):
    chars = "/—\|"
    for char in chars:
        sys.stdout.write('\r'+mes+char)
        time.sleep(.5)
        sys.stdout.flush()


def change_model(mod="hog"):
    global m
    m = mod


def run(mod="hog", p=False):
    if p is False:
        print("[INFO] Running facial encoder over face database...")
    change_model(mod)
    print("[INFO] Model in use:", "Histogram of Oriented Gradient" if m is "hog" else "Convolution Neural Network")
    imagePaths = list(paths.list_images("images"))
    knownEncodings = []
    knownNames = []

    # loop over the image paths
    for (i, imagePath) in enumerate(imagePaths):
        # extract the person name from the image path
        if p is False:
            print("[INFO] processing image {}/{}".format(i + 1, len(imagePaths)))
        name = imagePath.split(os.path.sep)[-2]

        # load the input image and convert it from BGR (OpenCV ordering)
        # to dlib ordering (RGB)
        image = cv2.imread(imagePath)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # detect the (x, y)-coordinates of the bounding boxes
        # corresponding to each face in the input image
        boxes = face_recognition.face_locations(rgb, model=m)

        # compute the facial embedding for the face
        encodings = face_recognition.face_encodings(rgb, boxes, num_jitters=1)

        # loop over the encodings
        for encoding in encodings:
            # add each encoding + name to our set of known names and
            # encodings
            knownEncodings.append(encoding)
            knownNames.append(name)

    # dump the facial encodings + names to disk
    if p is False:
        print("[INFO] serializing encodings...")
    data = {"encodings": knownEncodings, "names": knownNames}
    f = open("encodings_" + m + ".pkl", "wb")
    f.write(pickle.dumps(data))
    f.close()
    return "DONE"


if __name__ == "__main__":
    run()
