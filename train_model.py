from ultralytics import YOLO

model = YOLO("yolo11n.pt")
results = model.train(data="dataset/data.yaml", epochs=70, imgsz=640)