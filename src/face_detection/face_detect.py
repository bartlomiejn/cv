import cv2
import numpy as np
from argparse import ArgumentParser

ap = ArgumentParser()
ap.add_argument(
    "-c", 
    "--classifier", 
    type=str, 
    help="Cascade classifier file")
args = vars(ap.parse_args())

process_each = 5
current = 0
cap = cv2.VideoCapture(0)
casc = cv2.CascadeClassifier(args["classifier"])

if not cap.isOpened():
    print("Camera open failure")
    exit(-1)

if casc.empty():
        print("Failed to load classifier")
        exit(-1)

while True:
    err, frame = cap.read()

    if not err:
        print(f"Did not receive frame, err: {err}")
        break

    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if current >= process_each:
        faces = casc.detectMultiScale(grayscale, 1.1, 4)

        for (x, y, w, h) in faces:
            cv2.rectangle(grayscale, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
        current = 0

    cv2.imshow('frame', grayscale)

    if cv2.waitKey(1) == ord('q'):
        break

    current += 1

cap.release()
cv2.destroyAllWindows()