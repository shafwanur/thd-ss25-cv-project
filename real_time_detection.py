import cv2
import math
from ultralytics import YOLO

# --- Our best pretrained model
model = YOLO("runs/detect/train2/weights/best.pt")  

# --- Capture camera feed
cap = cv2.VideoCapture(0)

# --- Set the camera resolution to 640x480
cap.set(3, 640)
cap.set(4, 480)

# --- Classnames the model can currently detect
classNames = ["chicken"]
# classNames = ["chicken", "egg"] # uncomment when the model is trained on eggs as well.

chicken_count = 0

while True:
    success, img = cap.read()
    results = model(img, stream=True)

    # --- Visual parameters for .putText() 
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (255, 0, 0)
    thickness = 1
    text = f"Chicken Count: {chicken_count}"
    (text_width, text_height), _ = cv2.getTextSize(text, font, fontScale, thickness)
    
    # --- Have a 10 pixel buffer from the top right corner
    x = img.shape[1] - text_width - 10 
    y = text_height + 10 

    # --- Display chicken count at the top right corner
    cv2.putText(img, text, (x, y), font, fontScale, color, thickness)

    for r in results:
        boxes = r.boxes
        chicken_count = len(boxes)

        for box in boxes:
            # --- Detect bounding boxes
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # --- Model confidence
            confidence = math.ceil((box.conf[0]*100))/100

            # --- Classname of the detected object
            cls = int(box.cls[0])

            # --- For visualisation purposes
            org = [x1, y1]

            # --- Only print out the bounding box and details about the object if we're at least 70% sure.
            if confidence > 0.70: 
              # --- Display the rectangular bounding box
              cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

              # --- Display the text we want to show
              cv2.putText(img, f"{classNames[cls]} {confidence}", org, font, fontScale, color, thickness)

    cv2.imshow('Webcam', img)

    # --- To break execution, press q on the keyboard
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()