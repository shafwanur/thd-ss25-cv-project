from ultralytics import YOLO

model = YOLO("yolo11n.pt")  # load a pretrained model
# Train the model
results = model.train(data="./data/desc.yaml", epochs=3, imgsz=640)
