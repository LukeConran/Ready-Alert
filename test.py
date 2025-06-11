import cv2

for i in range(5):  # Try camera indices 0 to 4
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera found at index {i}")
        cap.release()
        break
    else:
        print(f"No camera at index {i}")
else:
    print("Error: Could not open any camera.")