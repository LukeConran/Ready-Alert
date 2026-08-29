import sys
import json
import time
import threading
import cv2
import numpy as np
from flask import Flask, Response

sys.path.insert(0, 'detection')
import detector

detector.EAR_THRESHOLD = 0.8
detector.ALERT_PCT = 0.33
detector.WINDOW_FRAMES = 30

dlib_detector  = detector.detector
dlib_predictor = detector.predictor

app = Flask(__name__)


class CameraThread(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.lock = threading.Lock()
        self.latest_frame = None
        self.status = 'waiting'   # waiting | calibrating | alert | drowsy | no_face
        self.ear = None
        self._mode = 'waiting'    # waiting | calibrating | detecting
        self._calib_frames = []
        self._baseline = None
        self._window = []
        self._running = True
        self._calib_done = threading.Event()

    def start_calibration(self):
        self._calib_done.clear()
        with self.lock:
            self._calib_frames = []
            self._baseline = None
            self._window = []
            self._mode = 'calibrating'

    def wait_for_calibration(self):
        self._calib_done.wait()

    def run(self):
        vc = None
        for i in range(3):
            vc = cv2.VideoCapture(i)
            if vc.isOpened():
                break
            vc.release()

        while self._running:
            rval, frame = vc.read()
            if not rval:
                time.sleep(0.05)
                continue

            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = dlib_detector(gray, 1)
            ear = None

            if faces:
                lm = dlib_predictor(gray, faces[0])
                ear = detector._ear(lm, 36, 42) * 0.5 + detector._ear(lm, 42, 48) * 0.5

                # Draw eye outlines
                for s, e in [(36, 42), (42, 48)]:
                    pts = np.array([[lm.part(i).x, lm.part(i).y] for i in range(s, e)], np.int32)
                    cv2.polylines(frame, [pts], True, (0, 255, 0), 1)

            with self.lock:
                mode = self._mode

            if mode == 'calibrating':
                if ear is not None:
                    self._calib_frames.append(ear)
                if len(self._calib_frames) >= 90:
                    self._baseline = float(np.mean(self._calib_frames))
                    with self.lock:
                        self._mode = 'detecting'
                        self.status = 'alert'
                    self._calib_done.set()

            elif mode == 'detecting':
                if ear is None:
                    new_status = 'no_face'
                else:
                    threshold = self._baseline * detector.EAR_THRESHOLD
                    self._window.append(ear < threshold)
                    if len(self._window) > detector.WINDOW_FRAMES:
                        self._window.pop(0)
                    if len(self._window) == detector.WINDOW_FRAMES and np.mean(self._window) >= detector.ALERT_PCT:
                        new_status = 'drowsy'
                    else:
                        new_status = 'alert'

                with self.lock:
                    self.status = new_status
                    self.ear = round(ear, 3) if ear else None

            else:  # waiting or calibrating
                with self.lock:
                    self.status = mode
                    self.ear = None

            _, jpeg = cv2.imencode('.jpg', frame)
            with self.lock:
                self.latest_frame = jpeg.tobytes()

        vc.release()


cam = CameraThread()
cam.start()


def gen_frames():
    while True:
        with cam.lock:
            frame = cam.latest_frame
        if frame:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.033)


@app.route('/')
def index():
    return '''<!DOCTYPE html>
<html>
<head><title>Live Alert</title>
<style>
  body { font-family: Arial, sans-serif; max-width: 700px; margin: 40px auto; text-align: center; }
  #feed { width: 100%; border-radius: 8px; }
  #verdict { font-size: 2.5em; font-weight: bold; margin: 20px 0; min-height: 1.2em; }
  .drowsy { color: red; }
  .alert  { color: green; }
  .muted  { color: gray; font-size: 1em; }
  button  { padding: 12px 28px; font-size: 16px; cursor: pointer; border-radius: 6px; border: none; background: #333; color: #fff; }
</style>
</head>
<body>
  <h1>Live Alert</h1>
  <img id="feed" src="/feed">
  <div id="verdict" class="muted">—</div>

  <div id="calib-section">
    <p>We are going to simulate a car environment. As if you purchased the car for the first time,
    you will calibrate our device based on your alertness.<br><br>
    Act regularly alert for the next 3 seconds — not overdone, not sleepy, just as you normally would.</p>
    <button onclick="startCalib()">Begin Calibration</button>
  </div>

  <div id="detect-section" style="display:none">
    <p>We have stored your basic alertness. Now feel free to run the device.</p>
  </div>

<script>
let polling = false;

function startCalib() {
  document.getElementById('verdict').className = 'muted';
  document.getElementById('verdict').innerText = 'Calibrating...';
  document.querySelector('button').disabled = true;
  fetch('/calibrate').then(r => r.json()).then(data => {
    document.getElementById('calib-section').style.display = 'none';
    document.getElementById('detect-section').style.display = 'block';
    startPolling();
  });
}

function startPolling() {
  if (polling) return;
  polling = true;
  setInterval(() => {
    fetch('/status').then(r => r.json()).then(data => {
      const el = document.getElementById('verdict');
      if (data.status === 'drowsy') {
        el.className = 'drowsy'; el.innerText = 'DROWSY';
      } else if (data.status === 'alert') {
        el.className = 'alert'; el.innerText = 'NOT DROWSY';
      } else if (data.status === 'no_face') {
        el.className = 'muted'; el.innerText = 'No face detected';
      } else {
        el.className = 'muted'; el.innerText = '...';
      }
    });
  }, 500);
}
</script>
</body>
</html>'''


@app.route('/feed')
def feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/calibrate')
def calibrate_route():
    cam.start_calibration()
    cam.wait_for_calibration()
    return Response(json.dumps({'status': 'ok'}), mimetype='application/json')


@app.route('/status')
def status_route():
    with cam.lock:
        return Response(json.dumps({'status': cam.status, 'ear': cam.ear}), mimetype='application/json')


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5001, debug=False, threaded=True)
