import cv2
import os
import numpy as np
from tqdm import tqdm  # For progress bar

# Paths
data_path = "C:/Users/NISHANK/Desktop/miniproject/data"
masks_path = "C:/Users/NISHANK/Desktop/miniproject/masks"
annotated_path = "C:/Users/NISHANK/Desktop/miniproject/annotated_images"

# Ensure output directories exist
os.makedirs(masks_path, exist_ok=True)
os.makedirs(annotated_path, exist_ok=True)

# Function to process an individual image
def process_image(image_path, mask_output_path, annotated_output_path):
    # Read the image
    img = cv2.imread(image_path)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Strict thresholding to ignore dim light
    _, thresh = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)  # Adjusted threshold value to 230
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a mask
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)
    
    # Annotate the original image with contours
    annotated_img = img.copy()
    cv2.drawContours(annotated_img, contours, -1, (0, 255, 0), thickness=2)
    
    # Save the mask and annotated image
    cv2.imwrite(mask_output_path, mask)
    cv2.imwrite(annotated_output_path, annotated_img)

# Recursive function to process directories
def process_directory(input_dir, masks_dir, annotated_dir):
    for root, _, files in os.walk(input_dir):
        # Get relative path to maintain nested structure
        rel_path = os.path.relpath(root, input_dir)
        mask_subdir = os.path.join(masks_dir, rel_path)
        annotated_subdir = os.path.join(annotated_dir, rel_path)

        # Ensure subdirectories exist
        os.makedirs(mask_subdir, exist_ok=True)
        os.makedirs(annotated_subdir, exist_ok=True)

        for file in tqdm(files, desc=f"Processing {rel_path}", leave=False):
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                # Paths for input and output
                image_path = os.path.join(root, file)
                mask_output_path = os.path.join(mask_subdir, file)
                annotated_output_path = os.path.join(annotated_subdir, file)
                
                # Process the image
                process_image(image_path, mask_output_path, annotated_output_path)

# Process the dataset
process_directory(data_path, masks_path, annotated_path)

print("Processing complete. Masks and annotated images saved.")
