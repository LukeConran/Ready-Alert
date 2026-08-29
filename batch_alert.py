import sys
import json
import cv2
from flask import Flask, Response

sys.path.insert(0, 'detection')
import detector
from detector import calibrate, run

detector.EAR_THRESHOLD = 0.8
detector.ALERT_PCT = 0.33
detector.WINDOW_FRAMES = 30

app = Flask(__name__)
baseline = None


def capture_webcam(n=90):
    for index in range(3):
        vc = cv2.VideoCapture(index)
        if vc.isOpened():
            break
        vc.release()
    else:
        return None
    frames = []
    while len(frames) < n:
        rval, frame = vc.read()
        if not rval:
            break
        frames.append(frame)
    vc.release()
    return frames if frames else None


def json_response(data, status=200):
    return Response(json.dumps(data), status=status, mimetype='application/json')


@app.route('/')
def index():
    return '''<!DOCTYPE html>
<html>
<head><title>Ready Alert</title>
<style>
  body { font-family: Arial, sans-serif; max-width: 600px; margin: 60px auto; }
  button { padding: 12px 24px; font-size: 16px; cursor: pointer; }
  #result { font-size: 2em; font-weight: bold; margin-top: 30px; }
  .drowsy { color: red; }
  .alert  { color: green; }
</style>
</head>
<body>
  <h1>Ready Alert</h1>
  <div id="calibrate-section">
    <p>We are going to simulate a car environment. As if you purchased the car for the first time,
    you will calibrate our device based on your alertness. Act regularly alert for the next 3 seconds
    — not overdone, not sleepy, just as you normally would.</p>
    <button onclick="doCalibrate()">Begin Calibration</button>
    <p id="calib-status"></p>
  </div>
  <div id="detect-section" style="display:none">
    <p>We have stored your basic alertness. Now feel free to run the device.</p>
    <button onclick="doDetect()">Check Now</button>
    <div id="result"></div>
  </div>
<script>
function doCalibrate() {
  document.getElementById('calib-status').innerText = 'Calibrating... hold still and look alert (3s)';
  fetch('/calibrate').then(r => r.json()).then(data => {
    if (data.status === 'ok') {
      document.getElementById('calibrate-section').style.display = 'none';
      document.getElementById('detect-section').style.display = 'block';
    } else {
      document.getElementById('calib-status').innerText = 'Error: ' + data.message;
    }
  });
}
function doDetect() {
  document.getElementById('result').innerText = 'Checking...';
  fetch('/detect').then(r => r.json()).then(data => {
    if (data.error) {
      document.getElementById('result').innerText = 'Error: ' + data.error;
      return;
    }
    const el = document.getElementById('result');
    if (data.drowsy) {
      el.className = 'drowsy'; el.innerText = 'DROWSY';
    } else {
      el.className = 'alert'; el.innerText = 'NOT DROWSY';
    }
  });
}
</script>
</body>
</html>'''


@app.route('/calibrate')
def calibrate_route():
    global baseline
    frames = capture_webcam(90)
    if not frames:
        return json_response({'status': 'error', 'message': 'Could not open webcam'}, 500)
    baseline = calibrate(frames)
    if baseline is None:
        return json_response({'status': 'error', 'message': 'No face detected during calibration'}, 400)
    return json_response({'status': 'ok', 'baseline': round(baseline, 3)})


@app.route('/detect')
def detect_route():
    if baseline is None:
        return json_response({'error': 'Not calibrated yet'}, 400)
    frames = capture_webcam(90)
    if not frames:
        return json_response({'error': 'Could not open webcam'}, 500)
    is_drowsy = run(frames, baseline)
    return json_response({'drowsy': bool(is_drowsy)})


if __name__ == '__main__':
    app.run(debug=True)
