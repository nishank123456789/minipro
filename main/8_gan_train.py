import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# Paths
DATA_DIR = "C:/Users/NISHANK/Desktop/miniproject/data/train"
MASK_DIR = "C:/Users/NISHANK/Desktop/miniproject/UNET_image/masks/train"
INPAINTED_DIR = "C:/Users/NISHANK/Desktop/miniproject/inpainted/train"

# Generator: U-Net style
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1), nn.Tanh(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 1, 4, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Dataset loader
class InpaintDataset(Dataset):
    def __init__(self, data_dir, mask_dir, inpainted_dir, transform):
        self.data_dir = data_dir
        self.mask_dir = mask_dir
        self.inpainted_dir = inpainted_dir
        self.transform = transform
        self.samples = []

        for case_folder in os.listdir(data_dir):
            img_path = os.path.join(data_dir, case_folder)
            mask_path = os.path.join(mask_dir, case_folder)
            inp_path = os.path.join(inpainted_dir, case_folder)
            if not os.path.isdir(img_path): continue

            for fname in os.listdir(img_path):
                if fname.endswith('.png'):
                    self.samples.append((
                        os.path.join(img_path, fname),
                        os.path.join(mask_path, fname),
                        os.path.join(inp_path, fname)
                    ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path, inp_path = self.samples[idx]
        img = self.transform(Image.open(img_path).convert("RGB"))
        mask = self.transform(Image.open(mask_path).convert("L"))
        inp = self.transform(Image.open(inp_path).convert("RGB"))

        # Stack inpainted image + mask as input (4 channels)
        input_tensor = torch.cat([inp, mask], dim=0)
        return input_tensor, img

# Training function
def train():
    num_epochs = 20
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    dataset = InpaintDataset(DATA_DIR, MASK_DIR, INPAINTED_DIR, transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    G = Generator().to(device)
    D = Discriminator().to(device)

    criterion_GAN = nn.BCELoss()
    criterion_L1 = nn.L1Loss()
    optim_G = torch.optim.Adam(G.parameters(), lr=2e-4)
    optim_D = torch.optim.Adam(D.parameters(), lr=2e-4)

    for epoch in range(1, 21):
        loop = tqdm(dataloader, desc=f"Epoch {epoch}")
        for x, y in loop:
            x, y = x.to(device), y.to(device)
            pred_shape = D(y).shape
            valid = torch.ones(pred_shape, device=device)
            fake = torch.zeros(pred_shape, device=device)


            # Generator forward
            gen_img = G(x)
            loss_G = criterion_GAN(D(gen_img), valid) + 100 * criterion_L1(gen_img, y)
            optim_G.zero_grad()
            loss_G.backward()
            optim_G.step()

            # Discriminator forward
            real_loss = criterion_GAN(D(y), valid)
            fake_loss = criterion_GAN(D(gen_img.detach()), fake)
            loss_D = (real_loss + fake_loss) / 2
            optim_D.zero_grad()
            loss_D.backward()
            optim_D.step()

            loop.set_postfix(loss_G=loss_G.item(), loss_D=loss_D.item())

        if epoch + 1 == num_epochs:
            torch.save(G.state_dict(), "C:/Users/NISHANK/Desktop/miniproject/model/gan_generator.pth")


if __name__ == "__main__":
    train()
