import numpy as np
import dlib
import cv2

detector  = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('data/shape_predictor_68_face_landmarks.dat')

EAR_THRESHOLD = 0.85  # raised — drowsy dip is subtler than expected
WINDOW_FRAMES = 30    # rolling window size
ALERT_PCT     = 0.40  # alert if 40% of window is below threshold


def _ear(lm, s, e):
    pts = np.array([[lm.part(i).x, lm.part(i).y] for i in range(s, e)])
    A, B = np.linalg.norm(pts[1]-pts[5]), np.linalg.norm(pts[2]-pts[4])
    return (A + B) / (2.0 * np.linalg.norm(pts[0]-pts[3]))


def extract_ear(frame):
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 1)
    if not faces:
        return None
    lm = predictor(gray, faces[0])
    return (_ear(lm, 36, 42) + _ear(lm, 42, 48)) / 2.0


def calibrate(frames):
    ears = [extract_ear(f) for f in frames]
    return np.mean([e for e in ears if e is not None])


def run(frames, baseline):
    window = []
    for frame in frames:
        e = extract_ear(frame)
        if e is not None:
            window.append(e < baseline * EAR_THRESHOLD)
            if len(window) > WINDOW_FRAMES:
                window.pop(0)
            if len(window) == WINDOW_FRAMES and np.mean(window) >= ALERT_PCT:
                return True
    return False
