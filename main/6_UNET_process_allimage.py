import tensorflow as tf
import cv2
import numpy as np
import os
from tqdm import tqdm

# Paths
data_path = "C:/Users/NISHANK/Desktop/miniproject/data"
output_base_path = "C:/Users/NISHANK/Desktop/miniproject/UNET_image"
model_path = "C:/Users/NISHANK/Desktop/miniproject/model/unet_model_trained.keras"

# Load the trained model
model = tf.keras.models.load_model(model_path)
print("Model loaded successfully!")

# Function to preprocess input image
def preprocess_image(image_path, target_size=(256, 256)):
    image = cv2.imread(image_path)
    original_size = image.shape[:2]  # Save original dimensions for resizing later
    image_resized = cv2.resize(image, target_size)
    image_normalized = image_resized / 255.0  # Normalize to [0, 1]
    return np.expand_dims(image_normalized, axis=0), original_size

# Function to postprocess predicted mask
def postprocess_mask(predicted_mask, original_size):
    mask_resized = cv2.resize(predicted_mask, (original_size[1], original_size[0]))
    binary_mask = (mask_resized > 0.5).astype(np.uint8)  # Threshold to binary mask
    return binary_mask * 255  # Convert to [0, 255] for visualization

# Function to process directories
def process_directory(input_dir, output_dir_mask, output_dir_annotated):
    os.makedirs(output_dir_mask, exist_ok=True)
    os.makedirs(output_dir_annotated, exist_ok=True)

    for root, _, files in os.walk(input_dir):
        # Maintain directory structure in output
        relative_path = os.path.relpath(root, input_dir)
        mask_output_path = os.path.join(output_dir_mask, relative_path)
        annotated_output_path = os.path.join(output_dir_annotated, relative_path)

        os.makedirs(mask_output_path, exist_ok=True)
        os.makedirs(annotated_output_path, exist_ok=True)

        for file in tqdm(files, desc=f"Processing {relative_path}", leave=False):
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                image_path = os.path.join(root, file)
                mask_path = os.path.join(mask_output_path, file)
                annotated_path = os.path.join(annotated_output_path, file)

                # Process each image
                input_image, original_size = preprocess_image(image_path)
                predicted_mask = model.predict(input_image)[0, :, :, 0]
                binary_mask = postprocess_mask(predicted_mask, original_size)

                # Save the predicted mask
                cv2.imwrite(mask_path, binary_mask)

                # Overlay mask on original image with green color
                original_image = cv2.imread(image_path)
                green_mask = np.zeros_like(original_image)
                green_mask[:, :, 1] = binary_mask  # Set green channel based on the mask
                overlay = cv2.addWeighted(original_image, 0.7, green_mask, 0.3, 0)
                cv2.imwrite(annotated_path, overlay)

# Process train, test, and val directories
for dataset_type in ["train", "test", "val"]:
    input_dir = os.path.join(data_path, dataset_type)
    output_dir_mask = os.path.join(output_base_path, "masks", dataset_type)
    output_dir_annotated = os.path.join(output_base_path, "annotated_image", dataset_type)
    process_directory(input_dir, output_dir_mask, output_dir_annotated)

print("Processing complete. Masks and annotated images saved.")
