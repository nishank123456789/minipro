import tensorflow as tf
import cv2
import numpy as np
import os

# Paths
model_path = "C:/Users/NISHANK/Desktop/miniproject/model/unet_model_trained.keras"
new_image_path = "C:/Users/NISHANK/Desktop/miniproject/test/input/new_image.png"
output_mask_path = "C:/Users/NISHANK/Desktop/miniproject/test/output/new_image_mask.png"
overlay_output_path = "C:/Users/NISHANK/Desktop/miniproject/test/output/new_image_overlay.png"


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

# Predict mask for new image
input_image, original_size = preprocess_image(new_image_path)
predicted_mask = model.predict(input_image)[0, :, :, 0]  # Get single-channel mask
binary_mask = postprocess_mask(predicted_mask, original_size)

# Save the predicted mask
cv2.imwrite(output_mask_path, binary_mask)
print(f"Predicted mask saved at {output_mask_path}")

# Overlay mask on original image with green color
original_image = cv2.imread(new_image_path)
green_mask = np.zeros_like(original_image)  # Create a blank mask
green_mask[:, :, 1] = binary_mask  # Set green channel based on the mask

# Add the green overlay on the original image
overlay = cv2.addWeighted(original_image, 0.7, green_mask, 0.3, 0)  # Adjust weights for blending
cv2.imwrite(overlay_output_path, overlay)
print(f"Overlay image with green mask saved at {overlay_output_path}")