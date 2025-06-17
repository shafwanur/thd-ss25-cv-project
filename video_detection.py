import cv2
from ultralytics import YOLO

'''
Given a youtube video link, detect objects using the pretrained model, and output a .mp4 file with the results.
'''

# --- Loud the best pretrained model
model = YOLO("runs/detect/train2/weights/best.pt")

results = model.track("https://youtu.be/LNwODJXcvt4", show=True)  # Tracking with default tracker


file_name = "video single chicken" # upload a video showing a chicken somewhere and chug it into "tests" and have the name here
# Open video
cap = cv2.VideoCapture(f"tests/{file_name}.mp4")

# Save output
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(f"{file_name}_annotated.mp4", fourcc, cap.get(cv2.CAP_PROP_FPS),
                      (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO on the frame
    results = model(frame)

    # Get annotated frame
    annotated_frame = results[0].plot()

    # Write the annotated frame to the output video
    out.write(annotated_frame)

cap.release()
out.release()