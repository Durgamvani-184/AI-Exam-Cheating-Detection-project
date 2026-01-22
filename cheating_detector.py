import cv2
import mediapipe as mp

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

LEFT_EYE = [33]
RIGHT_EYE = [263]
NOSE_TIP = 1   # face center reference

def detect_cheating(frame):
    cheating = False
    status = "Normal"

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:

        # 🔴 Multiple faces
        if len(results.multi_face_landmarks) > 1:
            return True, "Multiple Faces Detected"

        landmarks = results.multi_face_landmarks[0].landmark

        left_eye = landmarks[LEFT_EYE[0]]
        right_eye = landmarks[RIGHT_EYE[0]]
        nose = landmarks[NOSE_TIP]

        # 🔴 Head turned left / right
        eye_diff = abs(left_eye.x - right_eye.x)
        if eye_diff > 0.05:
            cheating = True
            status = "Looking Away"

        # 🔴 Face moved too much from center
        if nose.x < 0.35 or nose.x > 0.65:
            cheating = True
            status = "Face Not Centered"

    return cheating, status

