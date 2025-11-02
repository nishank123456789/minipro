import cv2
import os
import random

# Paths
input_path = "C:/Users/NISHANK/Desktop/miniproject/images"
output_path = "C:/Users/NISHANK/Desktop/miniproject/augmented_images"
os.makedirs(output_path, exist_ok=True)

# Function to load images from folder
def load_images_from_folder(folder):
    images = []
    image_paths = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                full_path = os.path.join(root, file)
                img = cv2.imread(full_path)
                if img is not None:
                    images.append(img)
                    image_paths.append(full_path)
    return images, image_paths

# Function to crop an image slightly
def crop_image(image, crop_percentage=0.2):
    height, width = image.shape[:2]
    crop_h = int(height * crop_percentage)
    crop_w = int(width * crop_percentage)
    return image[crop_h:height - crop_h, crop_w:width - crop_w]

# Function for horizontal flip
def horizontal_flip(image):
    return cv2.flip(image, 1)

# Function for vertical flip
def vertical_flip(image):
    return cv2.flip(image, 0)

# Function for random rotation
def random_rotation(image):
    angle = random.uniform(-30, 30)
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, rotation_matrix, (width, height), borderMode=cv2.BORDER_REFLECT)

# Load all images
images, image_paths = load_images_from_folder(input_path)

# Perform augmentation
for image, image_path in zip(images, image_paths):
    if image is not None:
        # Crop the image
        cropped_image = crop_image(image)

        # Generate augmented images
        augmented_images = [
            horizontal_flip(cropped_image),
            vertical_flip(cropped_image),
            random_rotation(cropped_image)
        ]

        # Create corresponding subfolder structure
        relative_path = os.path.relpath(image_path, input_path)
        save_folder = os.path.join(output_path, os.path.dirname(relative_path))
        os.makedirs(save_folder, exist_ok=True)

        # Save augmented images
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        for j, aug_img in enumerate(augmented_images):
            save_path = os.path.join(save_folder, f"{base_name}_aug_{j}.png")
            cv2.imwrite(save_path, aug_img)

print(f"Augmented images saved in {output_path}.")
