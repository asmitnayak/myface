import os
import sys
import threading
import time
import face_recognition
import concurrent.futures
import multiprocessing
import pickle
import cv2
from imutils import paths
import face_encoding


def run(multi_proc=False):
    thread_num = str(threading.get_ident())
    if os.path.exists("encodings_cnn.pkl") is False:
        print("[INFO] creating CNN encodings...")
        p1 = multiprocessing.Process(target=face_encoding.run, args=("cnn", True))
        p1.start()
        # the_process = threading.Thread(name='process', target=face_encoding.run, args=["cnn", True])
        # the_process.start()
        time.sleep(5)
        while p1.is_alive():
            face_encoding.animated_loading()

    print()
    print("[INFO] Thread Number: ", thread_num)
    print("[INFO] loading encoding...")
    data = pickle.loads(open("encodings_cnn.temp.pkl", "rb").read())

    # load the input images and convert from BGR to RGB
    # TO-DO
    imagePath = list(paths.list_images("temp"))
    sys.stdout.write("[INFO] recognizing faces...\n")
    for iPath in imagePath:
        image = cv2.imread(iPath)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # detect the (x, y)-coordinates of the bounding boxes corresponding
        # to each face in the input image, then compute the facial embeddings
        # for each face

        f1 = concurrent.futures.ProcessPoolExecutor().submit(face_recognition.face_locations, rgb, model="cnn")
        # while True:
        #     if f1.done():
        #         break
        #     face_encoding.animated_loading("Recognizing faces...")
        # sys.stdout.flush()
        boxes = f1.result()     # face_recognition.face_locations(rgb, model="cnn")
        encodings = face_recognition.face_encodings(rgb, boxes)

        # initialize the list of names for each face detected
        names = []

        # loop over the facial embeddings
        for encoding in encodings:
            # attempt to match each face in the input image to our known encodings
            matches = face_recognition.compare_faces(data["encodings"], encoding)
            name = "Unknown"

            # check to see if we have found a match
            if True in matches:
                # find the indexes of all matched faces then initialize a
                # dictionary to count the total number of times each face
                # was matched
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}

                # loop over the matched indexes and maintain a count for
                # each recognized face face
                for i in matchedIdxs:
                    name = data["names"][i]
                    counts[name] = counts.get(name, 0) + 1

                # determine the recognized face with the largest number of
                # votes (note: in the event of an unlikely tie Python will
                # select first entry in the dictionary)
                name = max(counts, key=counts.get)

            # update the list of names
            names.append(name)

        # # loop over the recognized faces
        # for ((top, right, bottom, left), name) in zip(boxes, names):
        #     # draw the predicted face name on the image
        #     cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        #     y = top - 15 if top - 15 > 15 else top + 15
        #     cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
        #                 0.75, (0, 255, 0), 2)
        if (len(names) == 0 or names[0] is "Unknown") and multi_proc is False:
            # show the output image
            cv2.imshow("Image", image)
            cv2.waitKey(0)
            cv2.destroyWindow("Image")
        elif len(names) > 0 and names[0] is not "Unknown":
            fileName = iPath.split(os.path.sep)[-1].split(".")[0]
            path = "images/" + str(names[0]) + "/" + str(fileName) + ".png"
            os.replace(str(iPath), path)

    print("[INFO] creating new CNN encodings...")
    face_encoding.run("cnn", True)
    # p1 = multiprocessing.Process(target=face_encoding.run, args=("cnn", True))
    # p1.start()
    # # the_process = threading.Thread(name='process', target=face_encoding.run, args=["cnn", True])
    # # the_process.start()
    # if multi_proc is False:
    #     time.sleep(5)
    #     while p1.is_alive():
    #         face_encoding.animated_loading()
    # else:
    #     while p1.is_alive():
    #         pass

    print("[INFO] Advanced Facial Recognition finished...")


if __name__ == "__main__":
    run(multi_proc=False)
