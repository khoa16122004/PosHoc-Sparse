from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from matplotlib import cm
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from constant import DEFAULT_VAL_DIR, IMAGENET_FOLDER2_CLASSNAME, IMAGENET_PROMPT_PATH
from util import (
    ImageNetVal,
    compute_grid_scores,
    get_CLIP_model,
    get_OPENCLIP_model,
    get_SIGLIP_model,
    get_torchvision_model,
    get_ViT_model,
    compose_grid_image,
)
from wrapper import SIGLIPWrapper, VLModelWrapper, VisionModelWrapper, VisionViTModelWrapper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate whether explanation maps localize the correct cell inside each 2x2 grid."
    )
    parser.add_argument(
        "--grid-path",
        type=Path,
        required=True,
        help="Path to the random grid JSON generated from selected_1000.json.",
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
        "--output-dir",
        type=Path,
        default="evaluate_explain_outputs",
        help="Directory where result JSON and images are written.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of grids to evaluate.",
    )
    parser.add_argument(
        "--cell-size",
        type=int,
        default=224,
        help="Pixel size of each cell in the composed 2x2 grid.",
    )
    parser.add_argument(
        "--save-grid-images",
        action="store_true",
        help="Save each composed grid image.",
    )
    parser.add_argument(
        "--save-heatmaps",
        action="store_true",
        help="Save one jet-colored explanation heatmap per evaluated cell.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device to use.",
    )
    return parser.parse_args()


def load_grid_entries(grid_path: Path) -> list[dict[str, Any]]:
    with grid_path.open("r", encoding="utf-8") as file_obj:
        payload = json.load(file_obj)

    grids = payload.get("grids") if isinstance(payload, dict) else payload
    if not isinstance(grids, list):
        raise ValueError("Grid JSON must be a list or a dict with a 'grids' list.")

    normalized_grids: list[dict[str, Any]] = []
    for grid_index, grid in enumerate(grids):
        if isinstance(grid, dict):
            components = grid.get("components", [])
            grid_id = grid.get("grid_id", grid_index)
        else:
            components = grid
            grid_id = grid_index

        if not isinstance(components, list) or len(components) != 4:
            raise ValueError("Each grid must contain exactly 4 components.")

        normalized_grids.append({"grid_id": grid_id, "components": components})

    return normalized_grids


def load_model(args: argparse.Namespace, device: torch.device):
    # json prompt path if vlm mode
    with open(IMAGENET_PROMPT_PATH, 'r') as f:
        class_prompts = json.load(f)    
    with open(IMAGENET_FOLDER2_CLASSNAME, 'r') as f:
        folder_2_class_name = json.load(f)
    
    # get model_name
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

    dataset = ImageNetVal(args.val_dir, transform=spatial)
    if args.type in ["CLIP", "OPENCLIP", "SIGLIP"]:
        model.set_fodler_class(dataset.classes, folder_2_class_name)
        model.extract_class_text_features()

    model.set_posthoc_xai(args.method)
    return model, spatial




def evaluate_grid(
    model,
    spatial,
    grid_entry: dict[str, Any],
    device: torch.device,
    cell_size: int,
    save_grid_images: bool,
    save_heatmaps: bool,
    image_output_dir: Path,
) -> dict[str, Any]:
    components = grid_entry["components"]
    grid_id = int(grid_entry["grid_id"])
    grid_pil = compose_grid_image(components, cell_size=cell_size)
    grid_tensor = spatial(grid_pil).unsqueeze(0).to(device)

    with torch.enable_grad():
        logits = model.predict(grid_tensor)
        top1_prediction = int(logits.argmax(dim=1).item())

    if save_grid_images:
        grid_pil.save(image_output_dir / f"grid_{grid_id:04d}.jpg", quality=95)

    evaluations: list[dict[str, Any]] = []
    for cell_index, component in enumerate(components):
        target_class = int(component["original_class"])
        with torch.enable_grad():
            _, saliency = model.predict_and_map(grid_tensor, class_id=target_class)

        grid_side = int(len(components) ** 0.5)
        if grid_side * grid_side != len(components):
            raise ValueError("Grid evaluation expects a square number of components.")

        pooled_cell_size = saliency.shape[-1] // grid_side
        if pooled_cell_size <= 0:
            raise ValueError(
                f"Invalid pooled cell size derived from saliency shape {tuple(saliency.shape)}."
            )

        cell_scores = compute_grid_scores(saliency, single_shape=pooled_cell_size)[0]
        predicted_cell = int(cell_scores.argmax().item())
        target_score = float(cell_scores[cell_index].item())

        if save_heatmaps:
            sal = saliency[0].detach().cpu().numpy()
            plt.imshow(sal, cmap='jet')
            plt.axis('off')
            plt.savefig(
                image_output_dir / f"grid_{grid_id:04d}_cell_{cell_index}_class_{target_class}.png"

            )

        evaluations.append(
            {
                "cell_index": cell_index,
                "target_class": target_class,
                "component": component,
                "cell_scores": [float(score) for score in cell_scores.tolist()],
                "target_score": target_score,
                "predicted_cell": predicted_cell,
                "is_localized": predicted_cell == cell_index,
            }
        )

    localization_hits = sum(item["is_localized"] for item in evaluations)
    return {
        "grid_id": grid_id,
        "top1_prediction": top1_prediction,
        "localization_hits": localization_hits,
        "localization_rate": localization_hits / len(evaluations),
        "evaluations": evaluations,
    }


def main(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    output_dir = Path(args.output_dir) / args.type / args.model_name.replace("/", "_") / args.method
    image_output_dir = output_dir / "images"

    output_dir.mkdir(parents=True, exist_ok=True)
    image_output_dir.mkdir(parents=True, exist_ok=True)

    grids = load_grid_entries(args.grid_path)
    if args.limit is not None:
        grids = grids[: args.limit]

    model, spatial = load_model(args, device=device)

    results: list[dict[str, Any]] = []
    total_cell_evaluations = 0
    total_localized = 0

    for grid_entry in tqdm(grids, desc="Evaluating grids"):
        result = evaluate_grid(
            model=model,
            spatial=spatial,
            grid_entry=grid_entry,
            device=device,
            cell_size=args.cell_size,
            save_grid_images=args.save_grid_images,
            save_heatmaps=args.save_heatmaps,
            image_output_dir=image_output_dir,
        )
        results.append(result)
        total_cell_evaluations += len(result["evaluations"])
        total_localized += result["localization_hits"]

    summary = {
        "type": args.type,
        "model_name": args.model_name,
        "method": args.method,
        "grid_path": str(args.grid_path),
        "grid_count": len(results),
        "cell_evaluations": total_cell_evaluations,
        "localized_cells": total_localized,
        "localization_accuracy": (
            total_localized / total_cell_evaluations if total_cell_evaluations else 0.0
        ),
        "results": results,
    }

    output_path = output_dir / "grid_explain_scores.json"
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Evaluated grids: {len(results)}")
    print(f"Cell evaluations: {total_cell_evaluations}")
    print(f"Localized cells: {total_localized}")
    print(f"Localization accuracy: {summary['localization_accuracy']:.6f}")
    print(f"Saved results to: {output_path}")


if __name__ == "__main__":
    main(parse_args())
