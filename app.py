from flask import Flask, Response
import cv2
import random
import json

app = Flask(__name__)
app.config['PROPAGATE_EXCEPTIONS'] = True

frames = []

MAX_FRAMES = 60  # ~5 seconds at 30fps

def capture_webcam():
    global frames
    frames = []
    for index in range(3):
        vc = cv2.VideoCapture(index)
        if vc.isOpened():
            break
        vc.release()
    else:
        return False  # No webcam found

    rval, frame = vc.read()
    while rval and len(frames) < MAX_FRAMES:
        frames.append(frame)
        rval, frame = vc.read()

    vc.release()
    return len(frames) > 0

@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>DROWSY DRIVING DETECTOR</title>
        <style>
            .top-100-pixels {
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100px;
                background-color: #e7f1f9;
                z-index: -10;
            }
        </style>
        <link rel="icon" href="https://static-00.iconduck.com/assets.00/sleeping-face-emoji-2048x2048-x3gtr8b8.png">
    </head>
    <body>
        <div class="top-100-pixels"></div>
        <div style="margin-top: 35px;">
            <h1 style="font-family: Verdana; color:#373e43">DROWSY DRIVING DETECTOR</h1>
        </div>
        <div style="margin-top: 50px; font-family: Arial">
            <p>Press the button below to begin capturing facial footage. It will proceed for two seconds.</p>
        </div>
        <div class="container" style="display: flex">
            <div class="left-div">
                <button onclick="runWebcam()">Start Webcam</button>
            </div>
            <div style="font-family: Arial" class="right-div">
                <p id="cam_status"></p>
            </div>
        </div>
        <div>
            <p id="dro_stat" style="font-family: Arial"></p>
        </div>
        <script>
        function runWebcam() {
            document.getElementById('cam_status').innerText = 'Capturing video...';
            fetch('/run_script')
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'success') {
                        updateStatus('Video captured successfully! Processing...');
                        runDrowsy();
                    } else {
                        updateStatus('Error capturing video: ' + data.message);
                    }
                })
                .catch(error => {
                    console.error('Error:', error);
                    updateStatus('Error capturing video!');
                });
        }
        function updateStatus(message) {
            document.getElementById('cam_status').innerText = message;
        }
        function runDrowsy() {
            fetch('/run_drowsy')
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'success') {
                        updateStatus2('You are NOT DROWSY! Under 65 percent of frames indicated a drowsy face. Drive safe!');
                    } else {
                        updateStatus2('You are DROWSY! Over 65 percent of frames indicate a drowsy face. DO NOT DRIVE!');
                    }
                })
                .catch(error => {
                    console.error('Error:', error);
                    updateStatus2('Error processing drowsiness detection!');
                });
        }
        function updateStatus2(message) {
            document.getElementById('dro_stat').innerText = message;
        }
        </script>
    </body>
    </html>
    '''

@app.route('/run_script')
def run_script():
    try:
        success = capture_webcam()
        if success:
            return Response(json.dumps({'status': 'success', 'message': 'Captured {} frames'.format(len(frames))}), status=200, mimetype='application/json')
        else:
            return Response(json.dumps({'status': 'error', 'message': 'Failed to open webcam'}), status=500, mimetype='application/json')
    except Exception as e:
        return Response(json.dumps({'status': 'error', 'message': str(e)}), status=500, mimetype='application/json')

@app.route('/run_drowsy')
def run_drowsy():
    try:
        # Simulate drowsiness detection with random result
        is_drowsy = random.randint(1, 2) == 1
        if is_drowsy:
            return Response(json.dumps({'status': 'error', 'message': 'Drowsy detected'}), status=500, mimetype='application/json')
        else:
            return Response(json.dumps({'status': 'success', 'message': 'Not drowsy'}), status=200, mimetype='application/json')
    except Exception as e:
        return Response(json.dumps({'status': 'error', 'message': str(e)}), status=500, mimetype='application/json')

if __name__ == '__main__':
    app.run(debug=True)