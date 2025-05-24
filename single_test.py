from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")  
results = model("tests/test2.jpg") 
print(results)
results[0].show()
print(results[0].boxes)
