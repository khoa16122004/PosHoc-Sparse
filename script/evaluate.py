from __future__ import annotations

import argparse
from html import parser
import json
import sys
from pathlib import Path
from unittest import loader
from unittest import loader
from tqdm import tqdm
import torch
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from util import ImageNetVal, get_CLIP_model, get_OPENCLIP_model, get_torchvision_model
from wrapper import VisionModelWrapper, VLModelWrapper
from constant import DEFAULT_VAL_DIR, IMAGENET_PROMPT_PATH, IMAGENET_FOLDER2_CLASSNAME, CLIP_PARAMS, OPENCLIP_PARAMS
from torch.utils.data import DataLoader



def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a Vision/VLM model on ImageNet val and export correctly classified samples."
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
        choices=["torchvision", "CLIP", "OPENCLIP"],
    )

    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default="evaluator_outputs",
        help="Directory where the sample list and performance summary are written.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Validation batch size.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader worker count.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of validation images to evaluate.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device to use.",
    )
    return parser.parse_args()


def main(args):
    # Output dir: outputdir/type/model_name/class_name...
    output_dir = os.path.join(args.output_dir, args.type, args.model_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # json prompt path if vlm mode
    with open(IMAGENET_PROMPT_PATH, 'r') as f:
        class_prompts = json.load(f)    
    with open(IMAGENET_FOLDER2_CLASSNAME, 'r') as f:
        folder_2_class_name = json.load(f)
    
    # get model_name
    if args.type == "torchvision":
        model, spatial, normalize = get_torchvision_model(args.model_name)
        model = VisionModelWrapper(model, normalize)
        
    elif args.type == "CLIP":
        model, spatial, normalize = get_CLIP_model(args.model_name)
        model = VLModelWrapper(model, normalize, class_prompts)

        
    elif args.type == "OPENCLIP":
        model, spatial, normalize, tokenizer = get_OPENCLIP_model(args.model_name)
        model = VLModelWrapper(model, normalize, class_prompts, tokenizer)
    
    
    # Dataset and dataloader    
    dataset = ImageNetVal(args.val_dir, transform=spatial)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    folder_class_list = dataset.classes
    if args.type in ["CLIP", "OPENCLIP"]:
        model.set_fodler_class(folder_class_list, folder_2_class_name)
        model.extract_class_text_features()
    
    total_seen = 0
    total_correct = 0
    correct_samples: list[dict[str, int | str]] = []
    per_class_total = [0 for _ in dataset.classes]
    per_class_correct = [0 for _ in dataset.classes]

    with torch.no_grad():
        for imgs, targets, paths in tqdm(dataloader):
            imgs = imgs.to(torch.device('cuda'), non_blocking=True)
            targets = targets.to(torch.device('cuda'), non_blocking=True)
            logits = model.predict(imgs)
            predictions = logits.argmax(dim=1)
            matches = predictions.eq(targets)

            for target in targets.detach().cpu().tolist():
                per_class_total[target] += 1

            for batch_index, is_match in enumerate(matches.detach().cpu().tolist()):
                target = int(targets[batch_index].item())
                total_seen += 1
                if is_match:
                    total_correct += 1
                    per_class_correct[target] += 1
                    correct_samples.append(
                        {
                            "id": len(correct_samples),
                            "img_path": str(Path(paths[batch_index]).resolve()),
                            "original_class": target,
                            "folder_name": dataset.classes[target],
                        }
                    )

                if args.limit is not None and total_seen >= args.limit:
                    break

    accuracy = (total_correct / total_seen) if total_seen else 0.0
    per_class_accuracy = []
    for class_index, folder_name in enumerate(dataset.classes):
        class_total = per_class_total[class_index]
        if class_total == 0:
            continue
        per_class_accuracy.append(
            {
                "class_index": class_index,
                "folder_name": folder_name,
                "correct": per_class_correct[class_index],
                "total": class_total,
                "accuracy": per_class_correct[class_index] / class_total,
            }
        )

    output_dir = Path(output_dir)
    samples_output_path = output_dir / f"sample_attacks_{args.model_name}.json"
    performance_output_path = output_dir / f"val_performance_{args.model_name}.json"
    samples_output_path.write_text(json.dumps(correct_samples, indent=2), encoding="utf-8")
    performance_output_path.write_text(
        json.dumps(
            {
                "model_name": args.model_name,
                "type": args.type,
                "val_dir": str(args.val_dir),
                "evaluated_samples": total_seen,
                "correct_samples": total_correct,
                "top1_accuracy": accuracy,
                "batch_size": args.batch_size,
                "num_workers": args.num_workers,
                "limit": args.limit,
                "sample_file": str(samples_output_path),
                "label_mapping": "ImageFolder alphabetical folder index",
                "per_class_accuracy": per_class_accuracy,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Evaluated samples: {total_seen}")
    print(f"Correct samples: {total_correct}")
    print(f"Top-1 accuracy: {accuracy:.6f}")
    print(f"Saved correct samples to: {samples_output_path}")
    print(f"Saved validation performance to: {performance_output_path}")



            
    
    
    
    
    
    
    








if __name__ == "__main__":
	main(parse_args())
