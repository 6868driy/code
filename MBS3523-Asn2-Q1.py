import cv2
import face_recognition

known_face_encodings = []
known_face_names = []
known_face_scores = []
for i in range(3):
    face_image = face_recognition.load_image_file(f"known_person_{i+1}.jpg")
    face_encoding = face_recognition.face_encodings(face_image)[0]
    known_face_encodings.append(face_encoding)
    known_face_names.append(f"Known Person {i+1}")
    known_face_scores.append(0.9)
unknown_image = face_recognition.load_image_file("unknown_person.jpg")

face_locations = face_recognition.face_locations(unknown_image)
face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

    name = "Unknown Person"
    score = 0

    for i, match in enumerate(matches):
        if match:
            name = known_face_names[i]
            score = known_face_scores[i]

    cv2.rectangle(unknown_image, (left, top), (right, bottom), (0, 255, 0), 2)
    cv2.putText(unknown_image, f"{name} ({score:.2f})", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

cv2.imshow("Recognized Faces", unknown_image)
cv2.waitKey(0)
