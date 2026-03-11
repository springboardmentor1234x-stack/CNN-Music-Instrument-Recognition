import os
import random
import shutil

source_dir = "spectrograms/train"
output_dir = "dataset_split"

train_ratio = 0.7
val_ratio = 0.15

for label in os.listdir(source_dir):

    label_path = os.path.join(source_dir, label)

    if not os.path.isdir(label_path):
        continue

    images = os.listdir(label_path)
    random.shuffle(images)

    train_end = int(len(images) * train_ratio)
    val_end = int(len(images) * (train_ratio + val_ratio))

    train_images = images[:train_end]
    val_images = images[train_end:val_end]
    test_images = images[val_end:]

    for img in train_images:
        dst = os.path.join(output_dir, "train", label)
        os.makedirs(dst, exist_ok=True)
        shutil.copy(os.path.join(label_path, img), dst)

    for img in val_images:
        dst = os.path.join(output_dir, "validation", label)
        os.makedirs(dst, exist_ok=True)
        shutil.copy(os.path.join(label_path, img), dst)

    for img in test_images:
        dst = os.path.join(output_dir, "test", label)
        os.makedirs(dst, exist_ok=True)
        shutil.copy(os.path.join(label_path, img), dst)

print("Dataset split completed!")
