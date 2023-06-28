import cv2
import dlib
import numpy as np
from math import  hypot

cap = cv2.VideoCapture(0)

detect = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def mid_pt(p1,p2):  #to find the mid point b/w two pts in the eye cords
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

font = cv2.FONT_HERSHEY_SIMPLEX

def blink_ratio(points,face_landmark):
    left_pt = (face_landmark.part(points[0]).x, face_landmark.part(points[0]).y)
    right_pt = (face_landmark.part(points[3]).x, face_landmark.part(points[3]).y)

    horizontal_line = cv2.line(frame, left_pt, right_pt, (0, 255, 0), 2)  # green-0,255,0 thick-2

    top_pt = mid_pt(face_landmark.part(points[1]), face_landmark.part(points[2]))  # mid pt
    bot_pt = mid_pt(face_landmark.part(points[5]), face_landmark.part(points[4]))

    vir_line = cv2.line(frame, top_pt, bot_pt, (0, 255, 0), 2)

    horizontl_dist = hypot((left_pt[0] - right_pt[0]), (left_pt[1] - right_pt[1]))
    vir_dist = hypot((top_pt[0] - bot_pt[0]), (top_pt[1] - bot_pt[1]))

    distance = horizontl_dist / vir_dist
    return distance

while True:
    _,frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detect(gray)
    for face in faces:
        #x,y = face.left(),face.top();
        #x1,y1 = face.right(),face.bottom();

        landmark = predictor(gray,face)

        dist = blink_ratio([36,37,38,39,40,41],landmark)

        if(dist > 5.7):
            cv2.putText(frame, "blink",(50,150),font,3,(255,0,0))

        #print(landmark)
        #print(face)

    cv2.imshow("frame",frame)

    key = cv2.waitKey(1)
    if key == ord('q') :
        break
cap.release()
cv2.destroyAllWindows()

