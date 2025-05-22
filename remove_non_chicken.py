import pandas as pd
import os

image_dir = "open-images-v7/train/data"
file_ids = downloaded_images = {os.path.splitext(f)[0] for f in os.listdir(image_dir) if f.endswith(".jpg")}
df = pd.read_csv("open-images-v7/train/labels/detections.csv")
filtered_df = df[(df['LabelName'] == '/m/09b5t') & (df['ImageID'].isin(downloaded_images))]

filtered_df.to_csv("filtered_detections_train.csv", index=True)
