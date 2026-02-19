import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from Generator import Generator
from similarity_search import find_similar_images

device = "cuda" if torch.cuda.is_available() else "cpu"

G = Generator().to(device)
G.load_state_dict(torch.load("checkpoints/generator.pth", map_location=device))
G.eval()

transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def enhance_image(image_path):
    low_img = transform(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)

    refs = find_similar_images(image_path, k=3)

    ref_imgs = []
    for r in refs:
        ref_path = r[0]
        img = transform(Image.open(ref_path).convert("RGB")).unsqueeze(0).to(device)
        ref_imgs.append(img)

    with torch.no_grad():
        output = G(low_img, ref_imgs)

    output = output.squeeze(0).cpu()
    output = (output * 0.5 + 0.5).clamp(0,1)

    return output


if __name__ == "__main__":
    test_img = "lol/low/22.png"
    enhanced = enhance_image(test_img)

    enhanced_img = enhanced.permute(1,2,0).numpy()

    plt.imsave("enhanced_output.png", enhanced_img)
    print("Saved enhanced image as enhanced_output.png")

