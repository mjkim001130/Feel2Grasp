"""
Train AutoEncoder for Left/Right camera images.

Based on train_AE.ipynb, adapted for side cameras with smaller latent dim.

Usage:
    python train_side_encoder.py --camera left --latent_dim 16
    python train_side_encoder.py --camera right --latent_dim 16
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from lerobot.datasets.lerobot_dataset import LeRobotDataset


# ======================== Networks (same structure as train_AE.ipynb) ========================
class ConvEncoder(nn.Module):
    """Convolutional encoder - same architecture as front encoder."""
    def __init__(self, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU(inplace=True),    # 128 -> 64
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(inplace=True),   # 64 -> 32
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(inplace=True),  # 32 -> 16
            nn.Conv2d(128, 256, 4, 2, 1), nn.ReLU(inplace=True), # 16 -> 8
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.fc = nn.Linear(256 * 4 * 4, latent_dim)

    def forward(self, x):
        return self.fc(self.net(x).flatten(1))


class ConvDecoder(nn.Module):
    """Convolutional decoder - same architecture as train_AE.ipynb."""
    def __init__(self, latent_dim: int, out_hw):
        super().__init__()
        self.out_hw = out_hw
        self.fc = nn.Linear(latent_dim, 256 * 4 * 4)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 3, 4, 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        h = self.fc(z).view(z.size(0), 256, 4, 4)
        x_hat = self.net(h)
        if (x_hat.shape[-2], x_hat.shape[-1]) != self.out_hw:
            x_hat = F.interpolate(x_hat, size=self.out_hw, mode="bilinear", align_corners=False, antialias=True)
        return x_hat


class AutoEncoder(nn.Module):
    """AutoEncoder - same structure as train_AE.ipynb."""
    def __init__(self, latent_dim: int, out_hw):
        super().__init__()
        self.encoder = ConvEncoder(latent_dim)
        self.decoder = ConvDecoder(latent_dim, out_hw)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


# ======================== Dataset ========================
def resize_chw(img_chw: torch.Tensor, hw):
    """Resize CHW tensor to target size."""
    img = img_chw.unsqueeze(0)  # [1,C,H,W]
    img = F.interpolate(img, size=hw, mode="bilinear", align_corners=False, antialias=True)
    return img.squeeze(0)


class SideCameraDataset(torch.utils.data.Dataset):
    """Dataset for side camera images - adapted from FrontOnlyDataset in train_AE.ipynb."""
    def __init__(self, ds, image_key, resize_hw=(128, 128)):
        self.ds = ds
        self.image_key = image_key
        self.resize_hw = resize_hw

        # Verify key exists
        probe = self.ds[0]
        if self.image_key not in probe:
            raise KeyError(f"{self.image_key} not in keys: {list(probe.keys())}")

        print(f"Dataset loaded: {len(self.ds)} samples, image_key={self.image_key}")

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        s = self.ds[idx]
        img = torch.as_tensor(s[self.image_key])  # (3,H,W), float32
        img = img.clamp(0.0, 1.0)
        img = resize_chw(img, self.resize_hw)     # (3,128,128)
        return img


# ======================== Training ========================
def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load dataset
    print(f"Loading dataset from {args.repo_id}...")
    ds = LeRobotDataset(args.repo_id, revision=args.revision, video_backend="pyav")
    print(f"Dataset size: {len(ds)}")

    # Find image key for the camera
    image_key = f"observation.images.{args.camera}"
    print(f"Using image key: {image_key}")

    # Create dataset and dataloader
    resize_hw = (args.resize, args.resize)
    img_ds = SideCameraDataset(ds, image_key=image_key, resize_hw=resize_hw)
    loader = DataLoader(
        img_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )

    # Verify batch
    batch = next(iter(loader))
    print(f"Batch shape: {batch.shape}, dtype: {batch.dtype}, range: [{batch.min().item():.4f}, {batch.max().item():.4f}]")

    # Create model
    ae = AutoEncoder(args.latent_dim, resize_hw).to(device)
    print(f"Model parameters: {sum(p.numel() for p in ae.parameters()):,}")

    # Optimizer
    opt = torch.optim.AdamW(ae.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler('cuda', enabled=(device == "cuda"))

    # Training
    os.makedirs(args.save_dir, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    ae.train()
    for ep in range(1, args.epochs + 1):
        running = 0.0

        pbar = tqdm(loader, desc=f"Epoch {ep}/{args.epochs}", leave=True)
        for imgs in pbar:
            imgs = imgs.to(device, non_blocking=True)

            # Resize (redundant if already resized in dataset, but safe)
            imgs = F.interpolate(
                imgs,
                size=resize_hw,
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=(device == "cuda")):
                recon = ae(imgs)
                loss = F.mse_loss(recon, imgs)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            running += float(loss.item())
            pbar.set_postfix(mse=running / (pbar.n + 1))

        avg_loss = running / len(loader)
        print(f"epoch {ep:03d} | mse {avg_loss:.6f}")

    # Save encoder
    save_path = os.path.join(args.save_dir, f"{args.camera}_encoder.pt")
    torch.save(
        {
            "repo_id": args.repo_id,
            "revision": args.revision,
            "image_key": image_key,
            "resize_hw": resize_hw,
            "latent_dim": args.latent_dim,
            "camera": args.camera,
            "encoder_state_dict": ae.encoder.state_dict(),
        },
        save_path,
    )
    print(f"Saved: {save_path}")
    print(f"Training complete! Final loss: {avg_loss:.6f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", type=str, default="mjkim00/Feel2Grasp", help="HuggingFace dataset repo ID")
    parser.add_argument("--revision", type=str, default="main", help="Dataset revision")
    parser.add_argument("--camera", type=str, default="left", choices=["left", "right"], help="Camera to train encoder for")
    parser.add_argument("--latent_dim", type=int, default=16, help="Latent dimension (8-16 recommended)")
    parser.add_argument("--resize", type=int, default=128, help="Image resize dimension")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_dir", type=str, default="./ae_side_out")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
