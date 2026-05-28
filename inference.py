"""Model and utility definitions for HW4 PromptIR restoration."""

from __future__ import annotations

import argparse
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as TF


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


def seed_everything(seed: int) -> None:
    """Set random seeds for reproducible data splitting and initialization."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def list_images(folder: Path) -> List[Path]:
    """Return sorted image paths from a folder."""
    if not folder.exists():
        raise FileNotFoundError(f"Missing folder: {folder}")
    return sorted(
        path for path in folder.iterdir()
        if path.suffix.lower() in IMAGE_EXTENSIONS
    )


def load_rgb_image(path: Path) -> Image.Image:
    """Load an image as RGB."""
    return Image.open(path).convert("RGB")


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert PIL image to a float tensor in [0, 1]."""
    return TF.to_tensor(image)


def tensor_to_uint8_chw(tensor: torch.Tensor) -> np.ndarray:
    """Convert a CHW torch tensor in [0, 1] to uint8 CHW numpy array."""
    tensor = tensor.detach().cpu().clamp(0, 1)
    return (tensor.numpy() * 255.0).round().astype(np.uint8)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(param.numel() for param in model.parameters()
               if param.requires_grad)


def get_clean_filename(degraded_name: str) -> str:
    """Map a degraded training filename to the corresponding clean filename."""
    if degraded_name.startswith("rain-"):
        return degraded_name.replace("rain-", "rain_clean-")
    if degraded_name.startswith("snow-"):
        return degraded_name.replace("snow-", "snow_clean-")
    raise ValueError(f"Unknown training filename: {degraded_name}")


class PairedAugmentation:
    """Paired crop and geometric augmentation for restoration training."""

    def __init__(self, img_size: int, training: bool = True) -> None:
        self.img_size = img_size
        self.training = training

    def random_crop(
        self,
        degraded: Image.Image,
        clean: Image.Image,
    ) -> Tuple[Image.Image, Image.Image]:
        """Apply the same random crop to degraded and clean images."""
        width, height = degraded.size
        if width < self.img_size or height < self.img_size:
            new_width = max(width, self.img_size)
            new_height = max(height, self.img_size)
            degraded = TF.resize(
                degraded,
                [new_height, new_width],
                antialias=True,
            )
            clean = TF.resize(clean, [new_height, new_width], antialias=True)
            width, height = degraded.size

        top = random.randint(0, height - self.img_size)
        left = random.randint(0, width - self.img_size)
        degraded = TF.crop(degraded, top, left, self.img_size, self.img_size)
        clean = TF.crop(clean, top, left, self.img_size, self.img_size)
        return degraded, clean

    def center_crop_or_resize(
        self,
        degraded: Image.Image,
        clean: Image.Image,
    ) -> Tuple[Image.Image, Image.Image]:
        """Center crop validation images, resizing small images if needed."""
        width, height = degraded.size
        if min(width, height) < self.img_size:
            degraded = TF.resize(
                degraded,
                [self.img_size, self.img_size],
                antialias=True,
            )
            clean = TF.resize(
                clean,
                [self.img_size, self.img_size],
                antialias=True,
            )
        else:
            degraded = TF.center_crop(
                degraded,
                [self.img_size, self.img_size],
            )
            clean = TF.center_crop(clean, [self.img_size, self.img_size])
        return degraded, clean

    def __call__(
        self,
        degraded: Image.Image,
        clean: Image.Image,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.training:
            degraded, clean = self.random_crop(degraded, clean)
            if random.random() < 0.5:
                degraded = TF.hflip(degraded)
                clean = TF.hflip(clean)
            if random.random() < 0.5:
                degraded = TF.vflip(degraded)
                clean = TF.vflip(clean)
            if random.random() < 0.75:
                rotations = random.randint(0, 3)
                if rotations:
                    degraded = TF.rotate(degraded, angle=90 * rotations)
                    clean = TF.rotate(clean, angle=90 * rotations)
        else:
            degraded, clean = self.center_crop_or_resize(degraded, clean)

        return pil_to_tensor(degraded), pil_to_tensor(clean)


class RestorationDataset(Dataset):
    """Paired degraded/clean image dataset."""

    def __init__(
        self,
        samples: List[Dict[str, Path]],
        transform: Optional[PairedAugmentation] = None,
    ) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[index]
        degraded = load_rgb_image(sample["degraded_path"])
        clean = load_rgb_image(sample["clean_path"])

        if self.transform is not None:
            degraded_tensor, clean_tensor = self.transform(degraded, clean)
        else:
            degraded_tensor = pil_to_tensor(degraded)
            clean_tensor = pil_to_tensor(clean)

        return {
            "degraded": degraded_tensor,
            "clean": clean_tensor,
            "type": sample["type"],
        }


class TestRestorationDataset(Dataset):
    """Test dataset returning degraded images and filenames."""

    def __init__(self, image_dir: Path) -> None:
        self.image_paths = list_images(image_dir)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Dict[str, object]:
        path = self.image_paths[index]
        image = load_rgb_image(path)
        return {
            "degraded": pil_to_tensor(image),
            "filename": path.name,
            "size": image.size,
        }


class LayerNorm2d(nn.Module):
    """Layer normalization over channels for 2D feature maps."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        variance = (x - mean).pow(2).mean(dim=1, keepdim=True)
        normalized = (x - mean) / torch.sqrt(variance + self.eps)
        return normalized * self.weight + self.bias


class ChannelAttention(nn.Module):
    """Squeeze-and-excitation style channel attention."""

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.net(x)


class GatedDWFFN(nn.Module):
    """Depthwise gated feed-forward network."""

    def __init__(self, channels: int, expansion: float = 2.0) -> None:
        super().__init__()
        hidden = int(channels * expansion)
        self.project_in = nn.Conv2d(channels, hidden * 2, 1)
        self.dwconv = nn.Conv2d(
            hidden * 2,
            hidden * 2,
            3,
            padding=1,
            groups=hidden * 2,
        )
        self.project_out = nn.Conv2d(hidden, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        return self.project_out(F.gelu(x1) * x2)


class RestorationBlock(nn.Module):
    """Residual restoration block with attention and gated FFN."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm1 = LayerNorm2d(channels)
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.Conv2d(channels, channels, 1),
            nn.GELU(),
            ChannelAttention(channels),
        )
        self.norm2 = LayerNorm2d(channels)
        self.ffn = GatedDWFFN(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.conv(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class PromptBlock(nn.Module):
    """Prompt selection and fusion block inspired by PromptIR."""

    def __init__(self, channels: int, num_prompts: int = 8) -> None:
        super().__init__()
        self.prompts = nn.Parameter(
            torch.randn(num_prompts, channels, 1, 1) * 0.02,
        )
        self.selector = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 2, 1),
            nn.GELU(),
            nn.Conv2d(channels // 2, num_prompts, 1),
            nn.Softmax(dim=1),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = x.shape
        weights = self.selector(x).unsqueeze(2)
        prompts = self.prompts.unsqueeze(0)
        prompt = (weights * prompts).sum(dim=1)
        prompt = prompt.expand(batch_size, channels, height, width)
        return x + self.fusion(torch.cat([x, prompt], dim=1))


def make_stage(channels: int, depth: int) -> nn.Sequential:
    """Create a sequential stage of restoration blocks."""
    return nn.Sequential(*[RestorationBlock(channels) for _ in range(depth)])


class ImprovedPromptIR(nn.Module):
    """Compact PromptIR-inspired image restoration network."""

    def __init__(
        self,
        base_channels: int = 56,
        num_prompts: int = 8,
        blocks: Optional[List[int]] = None,
    ) -> None:
        super().__init__()
        if blocks is None:
            blocks = [2, 3, 4]

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4

        self.intro = nn.Conv2d(3, c1, 3, padding=1)
        self.enc1 = make_stage(c1, blocks[0])
        self.down1 = nn.Conv2d(c1, c2, 3, stride=2, padding=1)
        self.enc2 = make_stage(c2, blocks[1])
        self.down2 = nn.Conv2d(c2, c3, 3, stride=2, padding=1)
        self.bottleneck = nn.Sequential(
            make_stage(c3, blocks[2]),
            PromptBlock(c3, num_prompts),
            make_stage(c3, blocks[2]),
        )
        self.up2 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.prompt2 = PromptBlock(c2, num_prompts)
        self.dec2 = make_stage(c2, blocks[1])
        self.up1 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.prompt1 = PromptBlock(c1, num_prompts)
        self.dec1 = make_stage(c1, blocks[0])
        self.outro = nn.Conv2d(c1, 3, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        network_input = x
        x1 = self.enc1(self.intro(x))
        x2 = self.enc2(self.down1(x1))
        x3 = self.bottleneck(self.down2(x2))

        y2 = self.up2(x3)
        if y2.shape[-2:] != x2.shape[-2:]:
            y2 = F.interpolate(
                y2,
                size=x2.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        y2 = self.dec2(self.prompt2(y2 + x2))

        y1 = self.up1(y2)
        if y1.shape[-2:] != x1.shape[-2:]:
            y1 = F.interpolate(
                y1,
                size=x1.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        y1 = self.dec1(self.prompt1(y1 + x1))
        residual = self.outro(y1)
        return (network_input + residual).clamp(0, 1)


class CharbonnierLoss(nn.Module):
    """Charbonnier loss."""

    def __init__(self, eps: float = 1e-3) -> None:
        super().__init__()
        self.eps = eps

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        return torch.mean(torch.sqrt((pred - target) ** 2 + self.eps ** 2))


def ssim_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Differentiable SSIM loss."""
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    mu_x = F.avg_pool2d(pred, 3, 1, 1)
    mu_y = F.avg_pool2d(target, 3, 1, 1)
    sigma_x = F.avg_pool2d(pred * pred, 3, 1, 1) - mu_x * mu_x
    sigma_y = F.avg_pool2d(target * target, 3, 1, 1) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(pred * target, 3, 1, 1) - mu_x * mu_y

    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (
        (mu_x ** 2 + mu_y ** 2 + c1)
        * (sigma_x + sigma_y + c2)
        + 1e-8
    )
    ssim = numerator / denominator
    return 1.0 - ssim.mean()


def gradient_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L1 loss on horizontal and vertical image gradients."""
    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
    return F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)


def restoration_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    charbonnier: CharbonnierLoss,
) -> torch.Tensor:
    """Combined restoration objective optimized for PSNR and structure."""
    l1_loss = F.l1_loss(pred, target)
    charb_loss = charbonnier(pred, target)
    structure_loss = ssim_loss(pred, target)
    edge_loss = gradient_loss(pred, target)
    return l1_loss + 0.5 * charb_loss + 0.15 * structure_loss + 0.05 * edge_loss


def compute_psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    max_value: float = 1.0,
) -> torch.Tensor:
    """Compute average PSNR over a batch."""
    mse = F.mse_loss(
        pred.clamp(0, 1),
        target,
        reduction="none",
    ).mean(dim=(1, 2, 3))
    return 10.0 * torch.log10((max_value ** 2) / (mse + 1e-8))


class EMA:
    """Exponential moving average of floating-point model states."""

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow = {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
            if value.dtype.is_floating_point
        }
        self.backup: Dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Update EMA state from the current model."""
        for key, value in model.state_dict().items():
            if key in self.shadow:
                self.shadow[key].mul_(self.decay).add_(
                    value.detach(),
                    alpha=1.0 - self.decay,
                )

    def apply_shadow(self, model: nn.Module) -> None:
        """Temporarily apply EMA weights to the model."""
        self.backup = {}
        state = model.state_dict()
        for key in self.shadow:
            self.backup[key] = state[key].detach().clone()
            state[key].copy_(self.shadow[key])

    def restore(self, model: nn.Module) -> None:
        """Restore non-EMA weights after validation."""
        state = model.state_dict()
        for key, value in self.backup.items():
            state[key].copy_(value)
        self.backup = {}


def build_samples(
    train_degraded_dir: Path,
    train_clean_dir: Path,
) -> List[Dict[str, object]]:
    """Build paired degraded/clean samples from assignment folders."""
    all_pairs: List[Dict[str, object]] = []
    for degraded_path in list_images(train_degraded_dir):
        clean_path = train_clean_dir / get_clean_filename(degraded_path.name)
        if clean_path.exists():
            degradation_type = (
                "rain" if degraded_path.name.startswith("rain-") else "snow"
            )
            all_pairs.append({
                "degraded_path": degraded_path,
                "clean_path": clean_path,
                "type": degradation_type,
            })
    return all_pairs


def create_loaders(
    data_root: Path,
    img_size: int,
    val_size: int,
    batch_size: int,
    num_workers: int,
    val_ratio: float,
    seed: int,
) -> Tuple[DataLoader, DataLoader, int, int]:
    """Create train and validation dataloaders."""
    train_dir = data_root / "train"
    train_degraded_dir = train_dir / "degraded"
    train_clean_dir = train_dir / "clean"

    all_pairs = build_samples(train_degraded_dir, train_clean_dir)
    if not all_pairs:
        raise RuntimeError("No valid train/clean pairs found.")

    train_pairs, val_pairs = train_test_split(
        all_pairs,
        test_size=val_ratio,
        random_state=seed,
        shuffle=True,
        stratify=[sample["type"] for sample in all_pairs],
    )

    train_dataset = RestorationDataset(
        train_pairs,
        PairedAugmentation(img_size, training=True),
    )
    val_dataset = RestorationDataset(
        val_pairs,
        PairedAugmentation(val_size, training=False),
    )
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    return train_loader, val_loader, len(train_dataset), len(val_dataset)


def create_model(
    base_channels: int,
    num_prompts: int,
    num_blocks: List[int],
    device: torch.device,
) -> ImprovedPromptIR:
    """Build the restoration model."""
    model = ImprovedPromptIR(
        base_channels=base_channels,
        num_prompts=num_prompts,
        blocks=num_blocks,
    )
    return model.to(device)


def augment_x8(x: torch.Tensor) -> List[torch.Tensor]:
    """Generate x8 test-time augmentations."""
    return [
        x,
        torch.flip(x, dims=[-1]),
        torch.flip(x, dims=[-2]),
        torch.flip(x, dims=[-2, -1]),
        torch.rot90(x, 1, dims=[-2, -1]),
        torch.rot90(x, 2, dims=[-2, -1]),
        torch.rot90(x, 3, dims=[-2, -1]),
        torch.flip(torch.rot90(x, 1, dims=[-2, -1]), dims=[-1]),
    ]


def deaugment_x8(outputs: List[torch.Tensor]) -> List[torch.Tensor]:
    """Invert x8 test-time augmentations."""
    return [
        outputs[0],
        torch.flip(outputs[1], dims=[-1]),
        torch.flip(outputs[2], dims=[-2]),
        torch.flip(outputs[3], dims=[-2, -1]),
        torch.rot90(outputs[4], -1, dims=[-2, -1]),
        torch.rot90(outputs[5], -2, dims=[-2, -1]),
        torch.rot90(outputs[6], -3, dims=[-2, -1]),
        torch.rot90(
            torch.flip(outputs[7], dims=[-1]),
            -1,
            dims=[-2, -1],
        ),
    ]


@torch.no_grad()
def predict_with_tta(
    model: nn.Module,
    x: torch.Tensor,
    use_tta: bool,
) -> torch.Tensor:
    """Run inference with optional x8 test-time augmentation."""
    if not use_tta:
        return model(x).clamp(0, 1)

    augmented = augment_x8(x)
    outputs = [model(augmented_x).clamp(0, 1) for augmented_x in augmented]
    outputs = deaugment_x8(outputs)
    return torch.stack(outputs, dim=0).mean(dim=0).clamp(0, 1)


def load_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    device: torch.device,
    use_ema: bool,
) -> Dict[str, object]:
    """Load model weights, optionally replacing them with EMA weights."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if use_ema and "ema_shadow" in checkpoint:
        state = model.state_dict()
        for key, value in checkpoint["ema_shadow"].items():
            if key in state:
                state[key].copy_(value)

    model.to(device).eval()
    return checkpoint


def validate_submission(pred_npz_path: Path, expected_count: int) -> None:
    """Validate CodaBench submission format."""
    loaded = np.load(pred_npz_path)
    expected_keys = [f"{index}.png" for index in range(expected_count)]

    def sort_key(filename: str) -> int:
        return int(filename.split(".")[0])

    actual_keys = sorted(loaded.files, key=sort_key)
    missing = sorted(set(expected_keys) - set(actual_keys), key=sort_key)
    extra = sorted(set(actual_keys) - set(expected_keys), key=sort_key)

    print("Number of predictions:", len(actual_keys))
    print("First keys:", actual_keys[:5])
    print("Last keys:", actual_keys[-5:])
    print("Missing keys:", missing)
    print("Extra keys:", extra)

    all_valid = not missing and not extra
    for key in expected_keys:
        if key not in loaded:
            all_valid = False
            continue

        array = loaded[key]
        if array.ndim != 3 or array.shape[0] != 3:
            print("Wrong shape:", key, array.shape)
            all_valid = False
        if array.dtype != np.uint8:
            print("Wrong dtype:", key, array.dtype)
            all_valid = False
        if array.min() < 0 or array.max() > 255:
            print("Wrong range:", key, array.min(), array.max())
            all_valid = False

    message = "pred.npz is valid." if all_valid else "pred.npz has errors."
    print(message)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate HW4 restoration predictions as pred.npz.",
    )
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints_hw4_improved/best_promptir_improved.pth"),
    )
    parser.add_argument("--output", type=Path, default=Path("pred.npz"))
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--num-prompts", type=int, default=8)
    parser.add_argument("--num-blocks", type=int, nargs=3, default=[2, 3, 4])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-tta", action="store_true")
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--expected-count", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    """Run inference and save pred.npz."""
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = create_model(
        base_channels=args.base_channels,
        num_prompts=args.num_prompts,
        num_blocks=args.num_blocks,
        device=device,
    )
    checkpoint = load_checkpoint(
        checkpoint_path=args.checkpoint,
        model=model,
        device=device,
        use_ema=not args.no_ema,
    )

    test_dir = args.data_root / "test" / "degraded"
    test_dataset = TestRestorationDataset(test_dir)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    predictions = {}
    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference"):
            degraded = batch["degraded"].to(device)
            filenames = batch["filename"]
            restored = predict_with_tta(
                model=model,
                x=degraded,
                use_tta=not args.no_tta,
            )

            for index, filename in enumerate(filenames):
                predictions[filename] = tensor_to_uint8_chw(restored[index])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **predictions)

    print("Loaded checkpoint:", args.checkpoint)
    print("Best validation PSNR:", checkpoint.get("best_val_psnr"))
    print(f"Saved {len(predictions)} predictions to {args.output.resolve()}")

    if args.expected_count > 0:
        validate_submission(args.output, args.expected_count)


if __name__ == "__main__":
    main()
