import cv2
import mediapipe as mp

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

LEFT_EYE = [33, 133]
RIGHT_EYE = [362, 263]

def detect_cheating(frame):
    cheating = False
    status = "Normal"

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        if len(results.multi_face_landmarks) > 1:
            return True, "Multiple Faces Detected"

        landmarks = results.multi_face_landmarks[0].landmark
        left_eye = landmarks[LEFT_EYE[0]]
        right_eye = landmarks[RIGHT_EYE[0]]

        if abs(left_eye.x - right_eye.x) > 0.15:
            cheating = True
            status = "Looking Away"

    return cheating, status
