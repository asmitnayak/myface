import argparse
import copy
import imutils
from datetime import datetime
import face_recognition
import pickle
import time
import cv2
import alert
import threading
import multiprocessing
import face_encoding
import face_recog_img

writer = None
output = ""


def run():
    global writer, output

    # create the facial encodings with model="hog"
    print("[INFO] creating HOG encodings...")
    the_process = threading.Thread(name='process', target=face_encoding.run, args=["hog", True])
    the_process.start()
    time.sleep(2)
    while the_process.isAlive():
        face_encoding.animated_loading()
    # load the facial encodings from the *.pkl file
    print("")
    print("[INFO] loading encodings...")
    data = pickle.loads(open("encodings_hog.pkl", "rb").read())

    print("[INFO] starting video stream...")
    # vs = VideoStream(src=0).start()
    cap = cv2.VideoCapture(0)

    unknown_cnt = 0

    # loop over frames from the video file stream
    while True:
        # grab the frame from the threaded video stream
        ret, frame = cap.read()
        orgi = copy.deepcopy(frame)

        # convert the input frame from BGR to RGB then resize it to have
        # a width of 750px (to speedup processing)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = imutils.resize(frame, width=750)
        r = frame.shape[1] / float(rgb.shape[1])

        # detect the (x, y)-coordinates of the bounding boxes
        # corresponding to each face in the input frame, then compute
        # the facial embeddings for each face
        boxes = face_recognition.face_locations(rgb, model="hog")
        encodings = face_recognition.face_encodings(rgb, boxes)
        names = []
        unk = 0

        # loop over the facial embeddings
        for encoding in encodings:
            # attempt to match each face in the input image to our known
            # encodings
            matches = face_recognition.compare_faces(data["encodings"], encoding, tolerance=0.5)
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

                # determine the recognized face with the largest number
                # of votes (note: in the event of an unlikely tie Python
                # will select first entry in the dictionary)
                name = max(counts, key=counts.get)

            # update the list of names
            names.append(name)

            if "asmit" not in names:
                for who in names:
                    if who not in ["Unknown"]:
                        alert.alert_me(who)
                        pass
                    else:
                        unknown_cnt += 1

            # loop over the recognized faces
            for ((top, right, bottom, left), name) in zip(boxes, names):
                # rescale the face coordinates
                top = int(top * r)
                right = int(right * r)
                bottom = int(bottom * r)
                left = int(left * r)

                # draw the predicted face name on the image
                cv2.rectangle(frame, (left, top), (right, bottom),
                              (0, 255, 0), 2)
                if name is "Unknown":
                    nameNew = "temp/" + str(datetime.now()).replace(" ", "_").replace(":", "_") + ".png"
                    cv2.imwrite(nameNew, orgi[top - 70:bottom + 50, left - 50:right + 50])
                    multiprocessing.Process(target=face_recog_img.run, args=[True]).start()

                y = top - 15 if top - 15 > 15 else top + 15
                cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, (0, 255, 0), 2)

            # if the video writer is None *AND* we are supposed to write
            # the output video to disk initialize the writer
            if writer is None and output is not None:
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                writer = cv2.VideoWriter(output, fourcc, 20,
                                         (frame.shape[1], frame.shape[0]), True)
    
            # if the writer is not None, write the frame with recognized
            # faces to disk
            if writer is not None:
                writer.write(frame)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    alert.alert_me2(unknown_cnt)

    # do a bit of cleanup
    cv2.destroyAllWindows()
    cap.release()


if __name__ == "__main__":
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--write", type=str, help="path to input directory of faces + images")
    args = vars(ap.parse_args())
    output = args["write"]
    run()
