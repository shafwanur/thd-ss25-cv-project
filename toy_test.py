import pandas as pd
import os

image_dir = "toy_folder/validate/images"
file_ids = downloaded_images = {os.path.splitext(f)[0] for f in os.listdir(image_dir) if f.endswith(".jpg")}
df = pd.read_csv("filtered_detections_validate.csv")
filtered_df = df[(df['ImageID'].isin(downloaded_images))]

filtered_df.to_csv("toy_detections_test.csv", index=True)
