import pandas as pd
from sklearn.model_selection import train_test_split
import os
import numpy as np

# Paths to the data
metadata_path = 'D:\\PIBIC\\python\\skincancer\\data\\HAM10000_metadata.csv'
image_dir_1 = 'D:\\PIBIC\\python\\skincancer\\data\\HAM10000_images_part_1'
image_dir_2 = 'D:\\PIBIC\\python\\skincancer\\data\\HAM10000_images_part_2'

# Load the metadata file
metadata = pd.read_csv(metadata_path)

# Add the ".jpg" extension to the image IDs
metadata['image_file'] = metadata['image_id'] + ".jpg"

# Full path of images
metadata['image_path'] = metadata['image_file'].apply(
    lambda x: os.path.join(image_dir_1, x) if os.path.exists(os.path.join(image_dir_1, x))
    else os.path.join(image_dir_2, x)
)

# Verify that all images were mapped correctly
if not all(metadata['image_path'].apply(os.path.exists)):
    raise FileNotFoundError("Some images listed in the CSV were not found in the specified folders.")

# Map class names to integer labels
class_names = sorted(metadata['dx'].unique())
class_to_label = {cls_name: idx for idx, cls_name in enumerate(class_names)}
metadata['label'] = metadata['dx'].map(class_to_label)

# Stratified split based on classes (dx)
train_metadata, val_metadata = train_test_split(
    metadata, test_size=0.3, random_state=42, stratify=metadata['dx']
)

# Create the .txt files with image paths and labels
train_metadata[['image_path', 'label']].to_csv(
    "../res/train_files.txt", index=False, header=False, sep='\t'
)
val_metadata[['image_path', 'label']].to_csv(
    "../res/val_files.txt", index=False, header=False, sep='\t'
)

# Display statistics
print(f"Total training images: {len(train_metadata)}")
print(f"Total validation images: {len(val_metadata)}")

# Check class distribution in the sets
print("\nTraining set distribution:")
print(train_metadata['dx'].value_counts())

print("\nValidation set distribution:")
print(val_metadata['dx'].value_counts())

print("Files 'train_files.txt' and 'val_files.txt' created successfully!")
