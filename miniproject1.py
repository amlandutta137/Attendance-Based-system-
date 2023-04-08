import face_recognition
import cv2
import numpy as np
import csv
import os
from datetime import datetime
video_capture = cv2.VideoCapture(0)

tripty_image = face_recognition.load_image_file("photos/tripty.jpg")
tripty_encoding = face_recognition.face_encodings(tripty_image)[0]

arnab_image = face_recognition.load_image_file("photos/arnab.jpg")
arnab_encoding = face_recognition.face_encodings(arnab_image)[0]

amlan_image = face_recognition.load_image_file("photos/amlan.jpg")
amlan_encoding = face_recognition.face_encodings(amlan_image)[0]

subhadeep_image = face_recognition.load_image_file("photos/subhadeep.jpg")
subhadeep_encoding = face_recognition.face_encodings(subhadeep_image)[0]

debrajsir_image = face_recognition.load_image_file("photos/debrajsir.jpg")
debrajsir_encoding = face_recognition.face_encodings(debrajsir_image)[0]

soham_image = face_recognition.load_image_file("photos/soham.jpg")
soham_encoding = face_recognition.face_encodings(soham_image)[0]

known_face_encoding =[
    tripty_encoding,
    arnab_encoding,
    amlan_encoding,
    subhadeep_encoding,
    debrajsir_encoding,
    soham_encoding
]
    
known_faces_names =[
    "tripty",
    "arnab",
    "amlan",
    "subhadeep",
    "debrajsir",
    "soham"
]

students = known_faces_names.copy()

face_locations = []
face_encodings = []
face_names = []
s=True

now = datetime.now()
current_date = now.strftime("%d-%m-%y")

f = open(current_date+ '.csv','w+',newline = '')
lnwriter =csv.writer(f)

while True:
    _,frame = video_capture.read()
    small_frame = cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
    rgb_small_frame = small_frame[:,:,::-1]
    if s:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame,face_locations)
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encoding,face_encoding)
            name=""
            face_distance = face_recognition.face_distance(known_face_encoding,face_encoding)
            best_match_index = np.argmin(face_distance)
            if matches[best_match_index]:
                name = known_faces_names[best_match_index]

            face_names.append(name)
            if name in known_faces_names:
                if name in students:
                    students.remove(name)
                    print(students)
                    current_time = now.strftime("%H:%M:%S")
                    current_date = now.strftime("%d-%m-%y")
                    lnwriter.writerow([name, current_time, current_date])
    cv2.imshow("attendance system",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
f.close()