from ultralytics import YOLO

model = YOLO("runs/detect/train2/weights/best.pt")  
results = model("test_image.jpeg") 
results[0].show()
print(results[0].boxes)
