from ultralytics import YOLO

model = YOLO("yolo11n.pt")  # load a pretrained model
# Train the model
results = model.train(data="./data/desc.yaml", epochs=70, imgsz=640)
