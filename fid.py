import os
import json
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from torchmetrics.image.fid import FrechetInceptionDistance


# -----------------------------
# DATASET (REAL + CAPTIONS)
# -----------------------------
class CocoFIDDataset(torch.utils.data.Dataset):
    def __init__(self, path, tokenizer, image_size):
        with open(f'{path}/annotations/captions_val2014.json') as f:
            self.data = json.load(f)['annotations']

        self.image_dir = f"{path}/val2014"
        self.tokenizer = tokenizer

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        caption = item['caption']
        image_id = item['image_id']

        image_name = f"COCO_val2014_{image_id:012d}.jpg"
        image_path = os.path.join(self.image_dir, image_name)

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        tokens = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt"
        )

        return image, tokens.input_ids.squeeze(0)


# -----------------------------
# FID COMPUTATION
# -----------------------------
@torch.no_grad()
def compute_fid(dataloader, device, generate_batch):
    fid = FrechetInceptionDistance(feature=2048).to(device)

    for real_images, input_ids in tqdm(dataloader):

        real_images = real_images.to(device)
        input_ids = input_ids.to(device)

        # Generate images
        fake_images = generate_batch(input_ids)

        # Convert to uint8 [0,255]
        real_images = (real_images * 255).to(torch.uint8)
        fake_images = (fake_images * 255).to(torch.uint8)

        fid.update(real_images, real=True)
        fid.update(fake_images, real=False)

    return fid.compute().item()


# -----------------------------
# MAIN
# -----------------------------
def main():
    parser = argparse.ArgumentParser()

    # keep similar style as your IS script
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_samples", type=int, default=5000)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # -----------------------------
    # LOAD YOUR MODELS (reuse yours)
    # -----------------------------
    print("Loading models...")

    unet = UNet2DConditionModel.from_pretrained(
        os.path.join(CHECKPOINT_PATH, "unet"),
        use_safetensors=True
    ).to(device)

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="linear")

    checkpoint = torch.load(os.path.join(CHECKPOINT_PATH, "training_state.pth"), map_location=device)
    unet.load_state_dict(checkpoint['model_state_dict'])

    unet.eval()
    vae.eval()
    text_encoder.eval()

    print("Models loaded.")

    # -----------------------------
    # GENERATION WRAPPER
    # -----------------------------
    @torch.no_grad()
    def generate_batch_wrapper(input_ids):
        return generate_batch(input_ids)  # reuse your function


    # -----------------------------
    # DATA
    # -----------------------------
    dataset = CocoFIDDataset(args.data_path, tokenizer, args.image_size)

    # limit samples (important for speed)
    dataset = torch.utils.data.Subset(dataset, range(min(len(dataset), args.num_samples)))

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # -----------------------------
    # COMPUTE FID
    # -----------------------------
    fid_score = compute_fid(dataloader, device, generate_batch_wrapper)

    print(f"\n🔥 FID Score: {fid_score:.4f}")


if __name__ == "__main__":
    main()