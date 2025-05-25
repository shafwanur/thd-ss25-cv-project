import sys
from pathlib import Path

import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

pt_path = Path(sys.argv[1])
image_path = Path(sys.argv[2]).absolute()

model = YOLO(pt_path)

results = model.predict(source=image_path)

for r in results:
    # https://stackoverflow.com/questions/75324341/yolov8-get-predicted-bounding-box#75332799
    annotator = Annotator(r.orig_img)
    boxes = r.boxes
    for box in boxes:
        b = box.xyxy[0]
        c = box.cls
        annotator.box_label(b, model.names[int(c)])
    img = annotator.result()

    relpath = Path(r.path).relative_to(image_path)
    save_name = Path("tmp/predicted") / relpath
    save_name.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(save_name, img)
