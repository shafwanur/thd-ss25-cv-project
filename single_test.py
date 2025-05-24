from ultralytics import YOLO

model = YOLO("runs/detect/train2/weights/best.pt")  
results = model("tests/test1.jpg") 
print(results)
results[0].show()
print(results[0].boxes)
