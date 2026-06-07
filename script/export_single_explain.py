from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from constant import DEFAULT_VAL_DIR, IMAGENET_FOLDER2_CLASSNAME, IMAGENET_PROMPT_PATH
from util import (
    ImageNetVal,
    get_CLIP_model,
    get_OPENCLIP_model,
    get_SIGLIP_model,
    get_torchvision_model,
    get_ViT_model,
    blend_overlay,
    saliency_to_jet_array,
    save_image_with_plt,
    save_tensor,
    tensor_to_rgb_array,
)
from wrapper import SIGLIPWrapper, VLModelWrapper, VisionModelWrapper, VisionViTModelWrapper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run post-hoc explanation on each image individually and save the spatial image, "
            "jet heatmap, and overlay."
        )
    )
    parser.add_argument(
        "--input-json",
        type=Path,
        required=True,
        help="Path to a sample list JSON. Each record should include img_path and optionally original_class.",
    )
    parser.add_argument(
        "--val-dir",
        type=Path,
        default=DEFAULT_VAL_DIR,
        help="ImageNet val directory organized as class folders.",
    )
    parser.add_argument(
        "--type",
        type=str,
        required=True,
        choices=["torchvision", "ViT", "CLIP", "OPENCLIP", "SIGLIP"],
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        help="Explanation method name supported by the selected wrapper.",
    )
    parser.add_argument(
        "--target-source",
        type=str,
        default="original_class",
        choices=["original_class", "predicted"],
        help="Which class to explain for each image.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default="single_explain_outputs",
        help="Directory where images and metadata are written.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of images to process.",
    )
    parser.add_argument(
        "--overlay-alpha",
        type=float,
        default=0.45,
        help="Heatmap opacity for overlay image.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device to use.",
    )
    return parser.parse_args()


def load_records(input_json: Path) -> list[dict[str, Any]]:
    with input_json.open("r", encoding="utf-8") as file_obj:
        payload = json.load(file_obj)

    if isinstance(payload, dict):
        records = payload.get("results") or payload.get("samples") or payload.get("grids")
    else:
        records = payload

    if not isinstance(records, list):
        raise ValueError("Input JSON must be a list of records or a dict containing a records list.")

    normalized_records: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError("Each input record must be a JSON object.")
        if "img_path" not in record:
            raise ValueError("Each input record must include img_path.")
        normalized = dict(record)
        normalized.setdefault("id", index)
        normalized_records.append(normalized)

    return normalized_records


def load_model(args: argparse.Namespace):
    with open(IMAGENET_PROMPT_PATH, "r", encoding="utf-8") as file_obj:
        class_prompts = json.load(file_obj)
    with open(IMAGENET_FOLDER2_CLASSNAME, "r", encoding="utf-8") as file_obj:
        folder_2_class_name = json.load(file_obj)

    if args.type == "torchvision":
        model, spatial, normalize = get_torchvision_model(args.model_name)
        model = VisionModelWrapper(model, normalize)
    elif args.type == "ViT":
        model, spatial, normalize = get_ViT_model(args.model_name)
        model = VisionViTModelWrapper(model, normalize)
    elif args.type == "CLIP":
        model, spatial, normalize, tokenizer = get_CLIP_model(args.model_name)
        model = VLModelWrapper(model, normalize, class_prompts, tokenizer)
    elif args.type == "OPENCLIP":
        model, spatial, normalize, tokenizer = get_OPENCLIP_model(args.model_name)
        model = VLModelWrapper(model, normalize, class_prompts, tokenizer)
    elif args.type == "SIGLIP":
        model, spatial, normalize, tokenizer = get_SIGLIP_model(args.model_name)
        model = SIGLIPWrapper(model, normalize, class_prompts, tokenizer)
    else:
        raise ValueError(f"Unsupported model type: {args.type}")

    dataset = ImageNetVal(args.val_dir, transform=spatial)
    if args.type in ["CLIP", "OPENCLIP", "SIGLIP"]:
        model.set_fodler_class(dataset.classes, folder_2_class_name)
        model.extract_class_text_features()

    model.set_posthoc_xai(args.method)
    return model, spatial


def export_single_record(
    model,
    spatial,
    record: dict[str, Any],
    device: torch.device,
    alpha: float,
    output_root: Path,
) -> dict[str, Any]:
    image_path = Path(record["img_path"])
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    with Image.open(image_path) as image:
        rgb_image = image.convert("RGB")
        image_tensor = spatial(rgb_image).unsqueeze(0).to(device)

    spatial_image = tensor_to_rgb_array(image_tensor[0])

    with torch.enable_grad():
        logits = model.predict(image_tensor)
        predicted_class = int(logits.argmax(dim=1).item())

    if record.get("target_class") is not None:
        explicit_target_class = int(record["target_class"])
    elif record.get("original_class") is not None:
        explicit_target_class = int(record["original_class"])
    else:
        explicit_target_class = predicted_class

    target_class = explicit_target_class
    with torch.enable_grad():
        _, saliency = model.predict_and_map(image_tensor, class_id=target_class)

    saliency_map = saliency[0]
    heatmap_image = saliency_to_jet_array(saliency_map)
    overlay_image = blend_overlay(spatial_image, heatmap_image, alpha=alpha)

    record_id = int(record["id"])
    stem = f"sample_{record_id:04d}_class_{target_class}"
    sample_dir = output_root / stem
    sample_dir.mkdir(parents=True, exist_ok=True)

    spatial_path = sample_dir / "input_spatial.png"
    saliency_tensor_path = sample_dir / "saliency_map.pt"
    heatmap_path = sample_dir / "explain_jet.png"
    overlay_path = sample_dir / "overlay.png"
    metadata_path = sample_dir / "metadata.json"

    save_image_with_plt(spatial_image, spatial_path)
    save_tensor(saliency_map, saliency_tensor_path)
    save_image_with_plt(heatmap_image, heatmap_path)
    save_image_with_plt(overlay_image, overlay_path)

    result = {
        "id": record_id,
        "img_path": str(image_path),
        "predicted_class": predicted_class,
        "target_class": target_class,
        "original_class": record.get("original_class"),
        "folder_name": record.get("folder_name"),
        "input_spatial_path": str(spatial_path),
        "saliency_tensor_path": str(saliency_tensor_path),
        "explain_jet_path": str(heatmap_path),
        "overlay_path": str(overlay_path),
    }
    metadata_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    output_root = Path(args.output_dir) / args.type / args.model_name.replace("/", "_") / args.method
    output_root.mkdir(parents=True, exist_ok=True)

    records = load_records(args.input_json)
    if args.limit is not None:
        records = records[: args.limit]

    model, spatial = load_model(args)

    if args.target_source == "predicted":
        for record in records:
            record.pop("target_class", None)
            record.pop("original_class", None)

    results: list[dict[str, Any]] = []
    for record in tqdm(records, desc="Exporting explanations"):
        results.append(
            export_single_record(
                model=model,
                spatial=spatial,
                record=record,
                device=device,
                alpha=args.overlay_alpha,
                output_root=output_root,
            )
        )

    summary_path = output_root / "summary.json"
    summary = {
        "type": args.type,
        "model_name": args.model_name,
        "method": args.method,
        "input_json": str(args.input_json),
        "target_source": args.target_source,
        "processed": len(results),
        "results": results,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Processed images: {len(results)}")
    print(f"Saved outputs to: {output_root}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main(parse_args())