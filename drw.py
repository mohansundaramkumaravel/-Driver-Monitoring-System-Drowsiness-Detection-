import cv2
import mediapipe as mp
import time
from math import hypot
import threading
from tkinter import Tk, messagebox

# -------------------------------
# MEDIA PIPE SETUP
# -------------------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# -------------------------------
# EYE LANDMARKS
# -------------------------------
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# -------------------------------
# PARAMETERS
# -------------------------------
EAR_THRESHOLD = 0.22          # Eye aspect ratio threshold
DROWSY_TIME = 2.5             # Seconds to trigger drowsy alert
SMOOTHING_FRAMES = 5          # Moving average smoothing
COUNTDOWN_TIME = 5            # Countdown before starting (seconds)

ear_history = []
eye_closed_start = None

# -------------------------------
# FUNCTIONS
# -------------------------------
def eye_aspect_ratio(eye_points):
    A = hypot(eye_points[1][0] - eye_points[5][0], eye_points[1][1] - eye_points[5][1])
    B = hypot(eye_points[2][0] - eye_points[4][0], eye_points[2][1] - eye_points[4][1])
    C = hypot(eye_points[0][0] - eye_points[3][0], eye_points[0][1] - eye_points[3][1])
    return (A + B) / (2.0 * C)

def send_alert_popup(reason):
    """Show alert in a popup window"""
    def popup():
        root = Tk()
        root.withdraw()  # Hide the main window
        messagebox.showwarning("Driver Alert", reason)
        root.destroy()
    # Run popup in a separate thread to avoid blocking the main loop
    threading.Thread(target=popup).start()

# -------------------------------
# CAMERA SETUP
# -------------------------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# -------------------------------
# COUNTDOWN BEFORE START
# -------------------------------
start_time = time.time()
while True:
    ret, frame = cap.read()
    if not ret:
        continue
    frame = cv2.flip(frame, 1)
    remaining = int(COUNTDOWN_TIME - (time.time() - start_time))
    if remaining <= 0:
        break
    cv2.putText(frame, f"Starting in {remaining} sec", (200, 360),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
    cv2.imshow("Driver Monitoring System", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        exit()

print("🚦 Driver Monitoring System Started")

# -------------------------------
# MAIN LOOP
# -------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    alert_color = (0, 255, 0)  # Green by default (alert)

    # -------------------------------
    # DROWSINESS DETECTION
    # -------------------------------
    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]

        left_eye = []
        right_eye = []

        for i in LEFT_EYE:
            x = int(face_landmarks.landmark[i].x * w)
            y = int(face_landmarks.landmark[i].y * h)
            left_eye.append((x, y))
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        for i in RIGHT_EYE:
            x = int(face_landmarks.landmark[i].x * w)
            y = int(face_landmarks.landmark[i].y * h)
            right_eye.append((x, y))
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        ear_left = eye_aspect_ratio(left_eye)
        ear_right = eye_aspect_ratio(right_eye)
        ear = (ear_left + ear_right) / 2.0

        # Smooth EAR
        ear_history.append(ear)
        if len(ear_history) > SMOOTHING_FRAMES:
            ear_history.pop(0)
        ear_smoothed = sum(ear_history) / len(ear_history)

        # Drowsiness logic
        if ear_smoothed < EAR_THRESHOLD:
            if eye_closed_start is None:
                eye_closed_start = time.time()
            elif time.time() - eye_closed_start > DROWSY_TIME:
                alert_color = (0, 0, 255)  # Red if drowsy
                cv2.putText(frame, "DROWSINESS ALERT!", (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, alert_color, 3)
                send_alert_popup("Driver Drowsy!")  # Popup alert
        else:
            eye_closed_start = None
            alert_color = (0, 255, 0)  # Green if alert

        # Draw EAR and status rectangle
        cv2.putText(frame, f"EAR: {ear_smoothed:.2f}", (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, alert_color, 2)
        cv2.rectangle(frame, (10, 10), (250, 120), alert_color, 3)

    # -------------------------------
    # DISPLAY FRAME
    # -------------------------------
    cv2.imshow("Driver Monitoring System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -------------------------------
# CLEANUP
# -------------------------------
cap.release()
cv2.destroyAllWindows()
print("🛑 System Stopped")
