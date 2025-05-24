'''
Filters relevant labels (currently, chicken labels) from all downloaded labels. 
The file directory structure presumes fiftyone was used to download the dataset without altering download.py.
'''

import os
import shutil
import pandas as pd

base_dir = r"C:\Users\shafw\fiftyone\open-images-v7" # location of the open-images-v7 folder

labels = pd.read_csv(base_dir + r"\train\labels\detections.csv") # all 1.9M labels

downloaded_imageIDs = [os.path.splitext(f)[0] for f in os.listdir(base_dir + r"\train\data") if f.endswith(".jpg")] # image IDs in open-images-v7/train/data

filtered_labels = labels[
    (labels['LabelName'] == '/m/09b5t') & # /m/09b5t identifies the "Chicken" class in open-images-v7
    (labels['ImageID'].isin(downloaded_imageIDs))
]

# Make a 80-20 split for training and validation
train_labels = filtered_labels.sample(frac=0.8, random_state=69)
val_labels = filtered_labels.drop(train_labels.index)

# Convert to CSV
os.makedirs("csv", exist_ok=True)
train_labels.to_csv("csv/train_detections.csv", index=True)
val_labels.to_csv("csv/val_detections.csv", index=True)

# Create the folder structure YOLO expects (if it doesn't already exist)
os.makedirs(os.path.join("dataset", "train", "images"), exist_ok=True)
os.makedirs(os.path.join("dataset", "train", "labels"), exist_ok=True)
os.makedirs(os.path.join("dataset", "val", "images"), exist_ok=True)
os.makedirs(os.path.join("dataset", "val", "labels"), exist_ok=True)

# Copy train images
train_image_ids = train_labels['ImageID'].unique()
for img_id in train_image_ids:
    src = os.path.join(base_dir, "train", "data", img_id + ".jpg")
    dst = os.path.join("dataset", "train", "images", img_id + ".jpg")
    if os.path.exists(src):
        shutil.copy2(src, dst)

# Copy val images
val_image_ids = val_labels['ImageID'].unique()
for img_id in val_image_ids:
    src = os.path.join(base_dir, "train", "data", img_id + ".jpg")
    dst = os.path.join("dataset", "val", "images", img_id + ".jpg")
    if os.path.exists(src):
        shutil.copy2(src, dst)


