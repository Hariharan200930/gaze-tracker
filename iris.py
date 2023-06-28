import cv2
import numpy as np
import dlib
import mediapipe as mp
import math
from math import hypot
mp_face_mesh = mp.solutions.face_mesh
left_eye =[362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398]
right_eye = [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246]
left_iris = [474,475,476,477]
right_iris = [469,470,471,472]

l_corner_l = [33]
l_corner_r = [133]
r_corner_l = [362]
r_corner_r = [263]



#v st

blinktimer = 0

detect = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def mid_pt(p1,p2):  #to find the mid point b/w two pts in the eye cords
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

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

#^ end
def eucli_distance(p1,p2):
    x1,y1 =p1.ravel()
    x2,y2 =p2.ravel()
    dist = math.sqrt((x2-x1)**2 + (y2-y1)**2)
    return dist

def iris_pos(iris_centre,r_pt,l_pt):
    cen_to_r_dist = eucli_distance(iris_centre,r_pt)
    tot_dist = eucli_distance(r_pt, l_pt)
    ratio = cen_to_r_dist/tot_dist
    iris_posi = ""
    if ratio>2.4 and ratio<2.7: #>2& <2.3 rt
        iris_posi = "right"
    elif ratio>2.7 and ratio<2.9: #>3 lt
        iris_posi = "left"
    else:
        iris_posi = "center"
    return iris_posi, ratio

cap = cv2.VideoCapture(0)
with mp_face_mesh.FaceMesh(max_num_faces=1,refine_landmarks = True,min_detection_confidence= 0.5, min_tracking_confidence=0.5)as face_mesh:
    while True:
        _,frame = cap.read()
        frame =cv2.flip(frame,1)

        rgb_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
#V st
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#^ end
        img_h,img_w = frame.shape[:2]

        result = face_mesh.process(rgb_frame)

        if result.multi_face_landmarks:
            #print(result.multi_face_landmarks[0].landmark)
            mesh_points = np.array([np.multiply((p.x,p.y),[img_w, img_h]).astype(int) for p in result.multi_face_landmarks[0].landmark])
            #print(mesh_points.shape)
            #cv2.polylines(frame, [mesh_points[left_eye]], True, (0, 255, 0), 1, cv2.LINE_AA)
            #cv2.polylines(frame, [mesh_points[right_eye]], True, (0, 255, 0), 1, cv2.LINE_AA)
            #cv2.polylines(frame, [mesh_points[left_iris]], True, (0, 255, 0), 1, cv2.LINE_AA)
            (l_cx, l_cy), l_radi = cv2.minEnclosingCircle(mesh_points[left_iris])
            (r_cx, r_cy), r_radi = cv2.minEnclosingCircle(mesh_points[right_iris])
            center_left = np.array([l_cx, l_cy],dtype=np.int32)
            center_right = np.array([r_cx, r_cy], dtype=np.int32)
            cv2.circle(frame, center_left,int(l_radi),(255,0,255),1,cv2.LINE_AA)
            cv2.circle(frame, center_right, int(r_radi), (255, 0, 255), 1, cv2.LINE_AA)
            #cv2.circle(frame, mesh_points[r_corner_l][0], 2, (255, 255, 255), 1, cv2.LINE_AA)
            #cv2.circle(frame, mesh_points[r_corner_r][0], 2, (0, 255, 255), 1, cv2.LINE_AA)
            iris_po,ratio =iris_pos(center_right, mesh_points[r_corner_r], mesh_points[r_corner_l][0])

            print(iris_po, ratio)          #ratio and iri pos

#v st


            faces = detect(gray)
            for face in faces:
                # x,y = face.left(),face.top();
                # x1,y1 = face.right(),face.bottom();

                landmark = predictor(gray, face)

                dist = blink_ratio([36, 37, 38, 39, 40, 41], landmark)

                if (dist > 5.7):
                    cv2.putText(frame, "blink", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0))

                    if blinktimer<3:
                        blinktimer+=1
                    else:
                        blinktimer=0

                    #print(blinktimer)


#^ end


        cv2.imshow("frame",frame)

        key = cv2.waitKey(1)

        if key == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()