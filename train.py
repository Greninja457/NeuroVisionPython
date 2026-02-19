import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from Generator import  Generator
from Discriminator import Discriminator
from similarity_search import find_similar_images

device = "cuda" if torch.cuda.is_available() else "cpu"

# Dataset

class LOLDataset(Dataset):
    def __init__(self, low_dir, high_dir):
        self.low_dir = low_dir
        self.high_dir = high_dir
        self.files = os.listdir(low_dir)

        self.transform = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]

        low_path = os.path.join(self.low_dir, file)
        high_path = os.path.join(self.high_dir, file)

        low_img = self.transform(Image.open(low_path).convert("RGB"))
        high_img = self.transform(Image.open(high_path).convert("RGB"))

        refs = find_similar_images(low_path, k=3)
        ref_imgs = []
        for r in refs:
            ref_path = r[0]
            img = self.transform(Image.open(ref_path).convert("RGB"))
            ref_imgs.append(img)

        return low_img, ref_imgs, high_img


# Setup

dataset = LOLDataset("lol/low", "lol/high")
loader = DataLoader(dataset, batch_size=4, shuffle=True)

G = Generator().to(device)
D = Discriminator().to(device)

criterion_gan = nn.BCEWithLogitsLoss()
criterion_l1 = nn.L1Loss()

optimizer_G = optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
optimizer_D = optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

epochs = 50


for epoch in range(epochs):
    for low_img, ref_imgs, high_img in loader:

        low_img = low_img.to(device)
        high_img = high_img.to(device)
        ref_imgs = [r.to(device) for r in ref_imgs]


        # Train Discriminator

        optimizer_D.zero_grad()

        fake_img = G(low_img, ref_imgs).detach()

        real_pred = D(high_img)
        fake_pred = D(fake_img)

        real_loss = criterion_gan(real_pred, torch.ones_like(real_pred))
        fake_loss = criterion_gan(fake_pred, torch.zeros_like(fake_pred))

        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()

        # Train Generator
        optimizer_G.zero_grad()

        fake_img = G(low_img, ref_imgs)
        pred = D(fake_img)

        gan_loss = criterion_gan(pred, torch.ones_like(pred))
        l1_loss = criterion_l1(fake_img, high_img)

        g_loss = gan_loss + 100 * l1_loss
        g_loss.backward()
        optimizer_G.step()

    print(f"Epoch [{epoch+1}/{epochs}] "
          f"D Loss: {d_loss.item():.4f} "
          f"G Loss: {g_loss.item():.4f}")

# Save Models

os.makedirs("checkpoints", exist_ok=True)

torch.save(G.state_dict(), "checkpoints/generator.pth")
torch.save(D.state_dict(), "checkpoints/discriminator.pth")

print("Training complete. Models saved.")
