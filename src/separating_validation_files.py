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

# Primeiro, separa em conjunto (treino + validação) e teste (15% para teste)
train_val_metadata, test_metadata = train_test_split(
    metadata, test_size=0.15, random_state=42, stratify=metadata['dx']
)

# Em seguida, divide o conjunto (treino + validação) em treino e validação (metade de 85% para cada)
# Isso resulta em aproximadamente 70% para treino e 15% para validação
train_metadata, val_metadata = train_test_split(
    train_val_metadata, test_size=0.5, random_state=42, stratify=train_val_metadata['dx']
)

# Cria os arquivos .txt com caminhos das imagens e rótulos
train_metadata[['image_path', 'label']].to_csv(
    "../res/train_files.txt", index=False, header=False, sep='\t'
)
val_metadata[['image_path', 'label']].to_csv(
    "../res/val_files.txt", index=False, header=False, sep='\t'
)
test_metadata[['image_path', 'label']].to_csv(
    "../res/test_files.txt", index=False, header=False, sep='\t'
)

# Exibe estatísticas
print(f"Total training images: {len(train_metadata)}")
print(f"Total validation images: {len(val_metadata)}")
print(f"Total test images: {len(test_metadata)}")

# Checa a distribuição das classes em cada conjunto
print("\nTraining set distribution:")
print(train_metadata['dx'].value_counts())

print("\nValidation set distribution:")
print(val_metadata['dx'].value_counts())

print("\nTest set distribution:")
print(test_metadata['dx'].value_counts())

print("Files 'train_files.txt', 'val_files.txt' and 'test_files.txt' created successfully!")
