#import opencv(cv2) library
import cv2

from random import randrange

#importing the algo

trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# img = cv2.imread('RDJ2.jpg')

# capture video images

webcam = cv2.VideoCapture(0)

#iterate forever on frames

while True:
    successful_frame_read, frame = webcam.read()

    #making the image grayscale
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detecting Faces
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    for (x,y,w,h) in face_coordinates:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (randrange(128,256),randrange(128,256),randrange(128,256)), 8)

    cv2.imshow('Face Detection', frame)
    key = cv2.waitKey(1)

    # stopping the video if q is pressed
    if key==81 or key==113:
        break

webcam.release()

print("Here is your Face Detection model")