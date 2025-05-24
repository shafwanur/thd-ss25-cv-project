'''
Converts open-images-v7 detection labels into a YOLO compatible format.
'''

import pandas as pd
import os

def convert(split): # split can be 'train' or 'val'
    df = pd.read_csv(f"csv/{split}_detections.csv")
    # Group by ImageName to handle multiple detections per image
    image_names = df['ImageID'].unique()
    output_dir = f"dataset/{split}/labels" # YOLO expects it in exactly this format.

    # Process each unique image
    for image_name in image_names:
        # Get all annotations for this image
        image_annotations = df[df['ImageID'] == image_name]
        
        # Create YOLO format annotation file (same name as image but .txt extension)
        base_name = os.path.splitext(image_name)[0]
        output_file = os.path.join(output_dir, f"{base_name}.txt")
        
        with open(output_file, 'w') as f:
            for _, ann in image_annotations.iterrows():
                # Get the already normalized coordinates
                x_min, x_max = float(ann['XMin']), float(ann['XMax'])
                y_min, y_max = float(ann['YMin']), float(ann['YMax'])
                
                # Convert to YOLO format (center coordinates + dimensions)
                x_center = (x_min + x_max) / 2
                y_center = (y_min + y_max) / 2
                width = x_max - x_min
                height = y_max - y_min
                
                # Class ID (0 for Chicken in this case)
                class_id = 0
                
                # Write to file
                f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")


convert("train")
convert("val")