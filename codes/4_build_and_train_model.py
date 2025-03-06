import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.utils import Sequence
import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Paths
train_images_path = "C:/Users/NISHANK/Desktop/miniproject/data/train"
train_masks_path = "C:/Users/NISHANK/Desktop/miniproject/masks/train"
val_images_path = "C:/Users/NISHANK/Desktop/miniproject/data/val"
val_masks_path = "C:/Users/NISHANK/Desktop/miniproject/masks/val"

# Data Generator for Segmentation
class SegmentationDataGenerator(Sequence):
    def __init__(self, image_dir, mask_dir, batch_size=16, target_size=(256, 256)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.batch_size = batch_size
        self.target_size = target_size
        self.image_filenames = []
        self.mask_filenames = []

        # Collect filenames
        for case_folder in sorted(os.listdir(image_dir)):
            case_images = sorted(os.listdir(os.path.join(image_dir, case_folder)))
            for img in case_images:
                self.image_filenames.append(os.path.join(image_dir, case_folder, img))
                self.mask_filenames.append(os.path.join(mask_dir, case_folder, img))

    def __len__(self):
        return len(self.image_filenames) // self.batch_size

    def __getitem__(self, idx):
        batch_images = self.image_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_masks = self.mask_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]

        images = []
        masks = []
        for img_file, mask_file in zip(batch_images, batch_masks):
            img = load_img(img_file, target_size=self.target_size)
            mask = load_img(mask_file, target_size=self.target_size, color_mode="grayscale")

            images.append(img_to_array(img) / 255.0)
            masks.append(img_to_array(mask) / 255.0)

        return np.array(images), np.array(masks)

# Prepare data generators
train_generator = SegmentationDataGenerator(train_images_path, train_masks_path, batch_size=8)
val_generator = SegmentationDataGenerator(val_images_path, val_masks_path, batch_size=8)

# Define U-Net Model
def build_unet(input_shape=(256, 256, 3)):
    inputs = Input(input_shape)

    # Encoder
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    p1 = MaxPooling2D((2, 2))(c1)
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    p2 = MaxPooling2D((2, 2))(c2)

    # Bottleneck
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(p2)

    # Decoder
    u1 = UpSampling2D((2, 2))(c3)
    d1 = Conv2D(128, (3, 3), activation='relu', padding='same')(u1)
    u2 = UpSampling2D((2, 2))(d1)
    d2 = Conv2D(64, (3, 3), activation='relu', padding='same')(u2)

    # Output layer
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(d2)

    return Model(inputs, outputs)

# Compile and Train the Model
model = build_unet()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_generator, validation_data=val_generator, epochs=10)
model.save("C:/Users/NISHANK/Desktop/miniproject/model/unet_model_trained.keras")

print("Model training complete and saved.")
