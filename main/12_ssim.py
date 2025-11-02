import matplotlib.pyplot as plt

# Example data: Replace these with your actual SSIM values
image_names = ["Image 1", "Image 2", "Image 3", "Image 4", "Image 5"]  # Add more as needed
ssim_values_unet = [0.90, 0.92, 0.91, 0.93, 0.92]  # Increased SSIM values for UNet
ssim_values_gan = [0.85, 0.88, 0.87, 0.89, 0.88]  # SSIM values for GAN

# Plot
plt.figure(figsize=(10, 6))
plt.plot(image_names, ssim_values_unet, marker='o', label="UNet SSIM", color="blue", linewidth=2)
plt.plot(image_names, ssim_values_gan, marker='s', label="GAN SSIM", color="green", linewidth=2)

# Customizing the chart
plt.ylabel("SSIM Values", fontsize=14)
plt.xlabel("Images", fontsize=14)
plt.title("SSIM Comparison Between UNet and GAN", fontsize=16)
plt.legend(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.ylim(0.8, 1.0)  # Focus on SSIM range

plt.tight_layout()
plt.show()
