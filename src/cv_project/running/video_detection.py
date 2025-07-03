import cv2
from ultralytics import YOLO
import math
import yt_dlp
import os

'''
Given a youtube video link, detect objects using the pretrained model, 
display the results in real-time, and output a .mp4 file with the results.
'''

# --- Download video
def download_video(video_url):
    filename = "video.mp4"

    # --- Formatting
    video_dl_options = {
        'format': 'bestvideo[height<=480][vcodec^=avc][ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        "outtmpl": filename
    }

    # --- If video doesn't exist, download it
    if not os.path.exists(filename):
        print(f"Downloading compatible video to: {filename}")
        with yt_dlp.YoutubeDL(video_dl_options) as ydl:
            ydl.download([video_url])
    else:
        print(f"Compatible video already exists at: {filename}")

    return filename

# --- Load the best pretrained model
model = YOLO("runs/detect/train3/weights/best.pt")

video_url = "https://www.youtube.com/watch?v=70IqKloH-mw&pp=ygUNY2hpY2tlbiB2aWRlbw%3D%3D"
video_path = download_video(video_url)

# --- Open video
cap = cv2.VideoCapture(video_path)
assert cap.isOpened(), f"Error reading video file at: {video_path}"

# --- Classnames the model can currently detect
classNames = ["chicken", "egg"]

# --- Video Writer
width, height, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
output_filename = "chicken_egg_detection.mp4"
video_writer = cv2.VideoWriter(output_filename,
                               cv2.VideoWriter_fourcc(*'avc1'), # 'avc1' is H.264, very compatible
                               fps,
                               (width, height))

font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.9
color = (0, 255, 0)
thickness = 2

while cap.isOpened():
    success, img = cap.read()
    if not success:
        print("End of video or error reading frame.")
        break
    
    results = model.track(img, persist=True, show=False, verbose=False)

    chicken_count = 0
    egg_count = 0

    # Get the boxes and track IDs
    # Check if there are any detections in the current frame
    if results[0].boxes is not None:
        boxes = results[0].boxes
        for box in boxes:
            confidence = math.ceil((box.conf[0] * 100)) / 100

            if confidence > 0.70: 
                # --- Detect bounding boxes
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                
                cls_index = int(box.cls[0])
                className = classNames[cls_index]
                
                label = f"{className} {confidence}"
                cv2.putText(img, label, (x1, y1 - 10), font, 0.8, color, thickness)
                
                if className == "chicken":
                    chicken_count += 1
                elif className == "egg":
                    egg_count += 1

    # --- Display counts on the frame (cleaner top-left placement) ---
    text_chicken = f"Chicken Count: {chicken_count}"
    text_eggs = f"Egg Count: {egg_count}"
    cv2.putText(img, text_chicken, (20, 40), font, fontScale, color, thickness)
    cv2.putText(img, text_eggs, (20, 80), font, fontScale, color, thickness)

    # --- To display the processed video in window
    cv2.imshow("YOLOv8 Live Detection", img)

    # --- Write the frame to the output file
    video_writer.write(img)

    # --- Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("Exiting...")
        break

# --- Kill everything on exit
cap.release()
video_writer.release()
cv2.destroyAllWindows()

print(f"Video processing complete. Output saved to: {output_filename}")