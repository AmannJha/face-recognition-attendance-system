import face_recognition
import cv2
import numpy as np
import csv 
import os
from datetime import datetime

# Load images
joti_image = face_recognition.load_image_file("C:/Users/91836/OneDrive/Desktop/Face_reconition majorr/photos/JyotiJhaPhoto (2) (1).jpg")
joti_face_detection = face_recognition.face_locations(joti_image)
if len(joti_face_detection) > 0:
    joti_face_detection = joti_face_detection[0]
else:
    print("No face detected in Joti's image. Exiting...")
    exit()

amanJha_image = face_recognition.load_image_file("C:/Users/91836/OneDrive/Desktop/Face_reconition majorr/photos/Aman Jha photo 1.jpg")
amanJha_face_detection = face_recognition.face_locations(amanJha_image)
if len(amanJha_face_detection) > 0:
    amanJha_face_detection = amanJha_face_detection[0]
else:
    print("No face detected in Aman Jha's image. Exiting...")
    exit()

known_face_locations = [
    joti_face_detection,
    amanJha_face_detection
]

known_student_names = [
    "joti",
    "amanJha"
]

students = known_student_names.copy()

# Open webcam
video_capture = cv2.VideoCapture(0)

now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

csv_file_path = current_date + '.csv'
f = open(csv_file_path, 'w+', newline='')
lnwriter = csv.writer(f)

while True:
    _, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]

    face_locations = face_recognition.face_locations(rgb_small_frame)
    
    if len(face_locations) > 0:  # Check if face locations are detected
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_locations, face_encoding)
            name = "Unknown"

            if True in matches:
                first_match_index = matches.index(True)
                name = known_student_names[first_match_index]

            face_names.append(name)

            if name in known_student_names:
                if name in students:
                    students.remove(name)
                    print(students)
                    current_time = now.strftime("%H-%M-%S")
                    lnwriter.writerow([name, current_time])

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    cv2.imshow("Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
f.close()
