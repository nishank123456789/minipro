import os
import torch
from torchvision import transforms
from PIL import Image
from model import Generator  # assuming model.py has correct Generator
from tqdm import tqdm

# Paths
model_path = "C:/Users/NISHANK/Desktop/miniproject/model/gan_generator.pth"
input_dir = "C:/Users/NISHANK/Desktop/miniproject/data/test"
mask_dir = "C:/Users/NISHANK/Desktop/miniproject/UNET_image/masks/test"
output_dir = "C:/Users/NISHANK/Desktop/miniproject/gan_output"

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Generator().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Create output structure and process
for root, _, files in os.walk(input_dir):
    for file in tqdm(files, desc="Processing"):
        if not file.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        # Build paths
        rel_path = os.path.relpath(root, input_dir)
        input_path = os.path.join(root, file)
        mask_path = os.path.join(mask_dir, rel_path, file)
        save_path = os.path.join(output_dir, rel_path)
        os.makedirs(save_path, exist_ok=True)
        output_file = os.path.join(save_path, file)

        # Load image and mask
        img = Image.open(input_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # grayscale mask

        img_tensor = transform(img)
        mask_tensor = transform(mask)

        # Concatenate mask with RGB -> 4-channel input
        input_tensor = torch.cat((img_tensor, mask_tensor), dim=0).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor).cpu().squeeze(0)
        
        output_image = transforms.ToPILImage()(output.clamp(0, 1))
        output_image.save(output_file)
