import os
import shutil
from sklearn.model_selection import train_test_split

# Paths
augmented_images_path = "C:/Users/NISHANK/Desktop/miniproject/augmented_images"
data_path = "C:/Users/NISHANK/Desktop/miniproject/data"
train_path = os.path.join(data_path, "train")
val_path = os.path.join(data_path, "val")
test_path = os.path.join(data_path, "test")

# Create directories for train, val, and test
os.makedirs(train_path, exist_ok=True)
os.makedirs(val_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)

# Collect all image paths with their class folder
all_images = []
for root, _, files in os.walk(augmented_images_path):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            class_name = os.path.basename(root)  # Get the class folder name
            file_path = os.path.join(root, file)
            all_images.append((file_path, class_name))

# Group images by class
class_image_dict = {}
for file_path, class_name in all_images:
    if class_name not in class_image_dict:
        class_image_dict[class_name] = []
    class_image_dict[class_name].append(file_path)

# Split and organize data for each class
for class_name, images in class_image_dict.items():
    train_files, test_files = train_test_split(images, test_size=0.2, random_state=42)
    train_files, val_files = train_test_split(train_files, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2

    # Create subdirectories for each class in train, val, and test
    os.makedirs(os.path.join(train_path, class_name), exist_ok=True)
    os.makedirs(os.path.join(val_path, class_name), exist_ok=True)
    os.makedirs(os.path.join(test_path, class_name), exist_ok=True)

    # Copy files to respective directories
    for file in train_files:
        shutil.copy(file, os.path.join(train_path, class_name, os.path.basename(file)))
    for file in val_files:
        shutil.copy(file, os.path.join(val_path, class_name, os.path.basename(file)))
    for file in test_files:
        shutil.copy(file, os.path.join(test_path, class_name, os.path.basename(file)))

print("Data split complete!")
