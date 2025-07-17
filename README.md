# 🐣 ChickenEgg Classifier
The project accompanies our A computer vision application that detects **chickens and eggs** in images and videos. It can count eggs moving on a **conveyor belt**, and also associate each egg with the **nearest chicken**.

## Demo
<p align="center">
  <img src="assets/demo_chicken_egg.gif" width="45%" />
  <img src="assets/demo_conveyor.gif" width="45%" />
</p>

---

## How We Made It

We started with a pretrained YOLOv11-small model and fine-tuned it on about 4,000 images of chickens and eggs sourced from Google’s OpenImages dataset. The raw data was preprocessed by converting annotations to the YOLO format, extracting bounding boxes for relevant classes, and cleaning out noisy and grouped samples.

Training was done using standard YOLOv11 scripts, with some hyperparameter tuning to achieve acceptable performance. For the interface, we built a PyQt6-based GUI that supports webcam, video files, and YouTube URLs, and displays bounding boxes, labels, chicken-egg associations, and object counts in real time.

---

## The Logic

Each egg is assigned to the closest visible chicken in the frame using simple Euclidean distance between bounding boxes. This works in both still images and video. In **video mode with conveyor belt enabled**, eggs are detected and counted as they pass by. 

We of course trained the model on real images of chickens and eggs, but as can be seem, the demo with the toy chickens seemed to work pretty well too :)