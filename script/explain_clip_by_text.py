from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from constant import CLIP_PARAMS
from util import (
    get_CLIP_model,
    blend_overlay,
    saliency_to_jet_array,
    save_image_with_plt,
    tensor_to_rgb_array,
)
from wrapper import VLModelWrapper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a CLIP explanation map for one image-text pair."
    )
    parser.add_argument(
        "--image-path",
        type=Path,
        required=True,
        help="Path to the input image.",
    )
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Text prompt used by CLIP for explanation.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        choices=sorted(CLIP_PARAMS.keys()),
    )
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        help="Explanation method name supported by the selected wrapper.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default="saliency_result/single_clip",
        help="Directory where metadata and visualization files are written.",
    )
    parser.add_argument(
        "--overlay-alpha",
        type=float,
        default=0.45,
        help="Alpha used when blending the heatmap over the image.",
    )
    parser.add_argument(
        "--save-npy",
        action="store_true",
        help="Also save the raw saliency map as a .npy file.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device to use.",
    )
    return parser.parse_args()


def load_model(args: argparse.Namespace, device: torch.device) -> tuple[VLModelWrapper, object]:
    model, spatial, normalize, tokenizer = get_CLIP_model(args.model_name)
    wrapper = VLModelWrapper(model, normalize, class_prompts={}, tokenizer=tokenizer, device=str(device))
    wrapper.class_text_features = wrapper.text_encode([args.text]).to(device)
    wrapper.set_posthoc_xai(args.method)
    return wrapper, spatial


def load_image(image_path: Path, spatial) -> tuple[Image.Image, torch.Tensor]:
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    with Image.open(image_path) as image:
        image_rgb = image.convert("RGB")
        image_tensor = spatial(image_rgb).unsqueeze(0)
    return image_rgb, image_tensor


def build_output_stem(image_path: Path, text: str) -> str:
    safe_text = "_".join(text.strip().split())[:60] or "prompt"
    safe_text = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in safe_text)
    return f"{image_path.stem}_{safe_text}"


def explain_single_image(
    model: VLModelWrapper,
    spatial,
    image_path: Path,
    text: str,
    device: torch.device,
    output_dir: Path,
    overlay_alpha: float,
    save_npy: bool,
) -> Path:
    image_rgb, image_tensor = load_image(image_path, spatial)
    image_tensor = image_tensor.to(device)

    with torch.enable_grad():
        logits, saliency = model.predict_and_map(image_tensor, class_id=0)

    score = float(logits[0, 0].item())
    saliency_map = saliency[0].detach().cpu()
    input_rgb = tensor_to_rgb_array(image_tensor[0])
    heatmap_rgb = saliency_to_jet_array(saliency_map)
    overlay_rgb = blend_overlay(input_rgb, heatmap_rgb, alpha=overlay_alpha)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_stem = build_output_stem(image_path, text)

    input_path = output_dir / f"{output_stem}_input.png"
    heatmap_path = output_dir / f"{output_stem}_heatmap.png"
    overlay_path = output_dir / f"{output_stem}_overlay.png"
    metadata_path = output_dir / f"{output_stem}_metadata.json"

    image_rgb.save(input_path)
    save_image_with_plt(heatmap_rgb, heatmap_path)
    save_image_with_plt(overlay_rgb, overlay_path)

    if save_npy:
        np.save(output_dir / f"{output_stem}_saliency.npy", saliency_map.numpy())

    metadata = {
        "image_path": str(image_path),
        "text": text,
        "model_name": model.model.name_or_path,
        "method": model.method_name,
        "device": str(device),
        "score": score,
        "input_image": str(input_path),
        "heatmap_image": str(heatmap_path),
        "overlay_image": str(overlay_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return metadata_path


def main(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    output_dir = Path(args.output_dir) / args.model_name.replace("/", "_") / args.method
    model, spatial = load_model(args, device=device)
    metadata_path = explain_single_image(
        model=model,
        spatial=spatial,
        image_path=args.image_path,
        text=args.text,
        device=device,
        output_dir=output_dir,
        overlay_alpha=args.overlay_alpha,
        save_npy=args.save_npy,
    )

    print(f"Saved explanation metadata to: {metadata_path}")


if __name__ == "__main__":
    main(parse_args())
