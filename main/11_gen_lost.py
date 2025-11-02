import matplotlib.pyplot as plt

# Sample data (replace with actual values from your training logs)
epochs = range(1, 11)  # 10 epochs
generator_loss = [6.41, 5.85, 5.32, 4.87, 4.43, 4.01, 3.31, 3.01, 2.70, 2.0]  # Example values
discriminator_loss = [0.71, 0.68, 0.66, 0.64, 0.62, 0.60, 0.58, 0.57, 0.55, 0.54]  # Example values

# Plot
plt.figure(figsize=(8, 5))
plt.plot(epochs, generator_loss, label='Generator Loss', color='blue', marker='o', linewidth=2)
plt.plot(epochs, discriminator_loss, label='Discriminator Loss', color='red', marker='o', linewidth=2)

plt.title('GAN Loss Over 10 Epochs', fontsize=16)
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
