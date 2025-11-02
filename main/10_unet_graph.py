import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def plot_rgb_values_unet(data_dir, unet_output_dir):
    for subfolder in os.listdir(data_dir):
        data_subfolder_path = os.path.join(data_dir, subfolder)
        unet_output_subfolder_path = os.path.join(unet_output_dir, subfolder)

        if not os.path.isdir(data_subfolder_path):
            continue

        for filename in os.listdir(data_subfolder_path):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                original_image_path = os.path.join(data_subfolder_path, filename)
                unet_output_path = os.path.join(unet_output_subfolder_path, filename)

                if not os.path.exists(unet_output_path):
                    print(f"❌ U-Net output not found for {filename}. Skipping.")
                    continue

                # Load images
                original = cv2.imread(original_image_path)
                unet_output = cv2.imread(unet_output_path)

                # Convert to RGB
                original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
                unet_output_rgb = cv2.cvtColor(unet_output, cv2.COLOR_BGR2RGB)

                # Extract RGB values
                r_original, g_original, b_original = original_rgb[:, :, 0], original_rgb[:, :, 1], original_rgb[:, :, 2]
                r_unet, g_unet, b_unet = unet_output_rgb[:, :, 0], unet_output_rgb[:, :, 1], unet_output_rgb[:, :, 2]

                # Plot R, G, B values
                x_axis = np.arange(original.shape[1])

                plt.figure(figsize=(15, 10))
                plt.suptitle(f"RGB Channel Comparison: {filename}")

                # Red channel
                plt.subplot(3, 1, 1)
                plt.title("Red Channel")
                plt.plot(x_axis, np.mean(r_original, axis=0), label="Original", color="red")
                plt.plot(x_axis, np.mean(r_unet, axis=0), label="U-Net Output", color="darkred")
                plt.legend()
                plt.grid()

                # Green channel
                plt.subplot(3, 1, 2)
                plt.title("Green Channel")
                plt.plot(x_axis, np.mean(g_original, axis=0), label="Original", color="green")
                plt.plot(x_axis, np.mean(g_unet, axis=0), label="U-Net Output", color="darkgreen")
                plt.legend()
                plt.grid()

                # Blue channel
                plt.subplot(3, 1, 3)
                plt.title("Blue Channel")
                plt.plot(x_axis, np.mean(b_original, axis=0), label="Original", color="blue")
                plt.plot(x_axis, np.mean(b_unet, axis=0), label="U-Net Output", color="darkblue")
                plt.legend()
                plt.grid()

                plt.tight_layout()
                plt.show()


# Example usage
plot_rgb_values_unet(
    "C:/Users/NISHANK/Desktop/miniproject/data/test",
    "C:/Users/NISHANK/Desktop/miniproject/UNET_image/masks/test"
)
