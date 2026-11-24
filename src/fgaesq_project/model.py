from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch


INFERENCE_ROOT = Path(__file__).resolve().parents[2] / "FG-IAA" / "FGAesQ_Inference"
if str(INFERENCE_ROOT) not in sys.path:
    sys.path.insert(0, str(INFERENCE_ROOT))

from utils.DiffToken import DiffToken  # noqa: E402
from utils.FGAesQ import FGAesQ  # noqa: E402


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


def resolve_device(requested: str = "auto") -> torch.device:
    requested = requested.lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA was requested ({requested}), but this PyTorch installation "
            "does not report CUDA availability."
        )
    return torch.device(requested)


def _state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            candidate = checkpoint.get(key)
            if isinstance(candidate, dict):
                checkpoint = candidate
                break
    if not isinstance(checkpoint, dict) or not checkpoint:
        raise ValueError("Checkpoint does not contain a non-empty state dictionary.")
    if not all(isinstance(key, str) for key in checkpoint):
        raise ValueError("Checkpoint state dictionary keys must be strings.")
    return checkpoint


def load_model(checkpoint_path: Path, device: torch.device) -> FGAesQ:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"FGAesQ checkpoint not found: {checkpoint_path}")

    model = FGAesQ(pretrained_path=None)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = _state_dict(checkpoint)
    incompatible = model.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Checkpoint does not match the official FGAesQ model. "
            f"Missing keys: {incompatible.missing_keys[:5]}; "
            f"unexpected keys: {incompatible.unexpected_keys[:5]}"
        )
    return model.to(device).eval()


def preprocess_image(model: FGAesQ, image_path: Path) -> tuple[torch.Tensor, ...]:
    from PIL import Image

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        preprocessor = DiffToken(clip_model=model.clip_model, patch_selection="random")
        return preprocessor.process_image(image, is_train=False)


def infer_image(
    model: FGAesQ,
    image_path: Path,
    device: torch.device,
    *,
    return_features: bool = False,
) -> tuple[float, torch.Tensor | None]:
    patches, pos_embed, mask = preprocess_image(model, image_path)
    patches = patches.unsqueeze(0).to(device)
    pos_embed = pos_embed.unsqueeze(0).to(device)
    mask = mask.unsqueeze(0).to(device)

    with torch.inference_mode():
        if return_features:
            features, (_, scores) = model(patches, pos_embed, mask, flag=True)
            return float(scores.squeeze().item()), features.squeeze(0).detach().cpu()
        scores = model(patches, pos_embed, mask)
        return float(scores.squeeze().item()), None
