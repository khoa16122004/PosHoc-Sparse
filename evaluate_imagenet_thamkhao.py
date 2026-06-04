from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
	sys.path.insert(0, str(ROOT))

from attack.util import DEFAULT_CHECKPOINT_DIR, load_bcos_model, load_imagenet_categories


IMAGENET_CATEGORIES = load_imagenet_categories()


class ImageFolderWithPath(ImageFolder):
	def __getitem__(self, index: int):
		sample, target = super().__getitem__(index)
		path, _ = self.samples[index]
		return sample, target, path


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Evaluate a B-cos model on ImageNet val and export correctly classified samples."
	)
	parser.add_argument(
		"--val-dir",
		type=Path,
		default=Path(r"E:\ImageNet1K\imagenet\ImageNet1K\val"),
		help="ImageNet val directory organized as class folders.",
	)
	parser.add_argument(
		"--checkpoint",
		type=Path,
		default=None,
		help="Optional explicit checkpoint path.",
	)
	parser.add_argument(
		"--model-name",
		type=str,
		default="resnet50",
		help="B-cos model name to load.",
	)
	parser.add_argument(
		"--checkpoint-dir",
		type=Path,
		default=DEFAULT_CHECKPOINT_DIR,
		help="Directory containing B-cos checkpoints.",
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=ROOT / "script" / "evaluator_outputs",
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


def normalize_checkpoint_dir(checkpoint_dir: Path) -> Path:
	checkpoint_dir = Path(checkpoint_dir)
	if checkpoint_dir.is_dir() and any(checkpoint_dir.glob("*.pth")):
		return checkpoint_dir
	bcos_subdir = checkpoint_dir / "bcos-v2"
	if bcos_subdir.is_dir():
		return bcos_subdir
	return checkpoint_dir


def category_name(index: int) -> str:
	if 0 <= index < len(IMAGENET_CATEGORIES):
		return IMAGENET_CATEGORIES[index]
	return str(index)


def main(args: argparse.Namespace) -> None:
	device = torch.device(args.device)
	checkpoint_dir = normalize_checkpoint_dir(args.checkpoint_dir)
	output_dir = args.output_dir / args.model_name
	output_dir.mkdir(parents=True, exist_ok=True)

	model_source = args.checkpoint if args.checkpoint is not None else args.model_name
	model, resolved_checkpoint_path = load_bcos_model(
		model_source,
		device=device,
		checkpoint_dir=checkpoint_dir,
		return_checkpoint_path=True,
	)
	transform_cls = model.transform

	dataset = ImageFolderWithPath(args.val_dir, transform=transform_cls.spatial_transform)
	loader = DataLoader(
		dataset,
		batch_size=args.batch_size,
		shuffle=False,
		num_workers=args.num_workers,
		pin_memory=device.type == "cuda",
	)

	total_seen = 0
	total_correct = 0
	correct_samples: list[dict[str, int | str]] = []
	per_class_total = [0 for _ in dataset.classes]
	per_class_correct = [0 for _ in dataset.classes]

	with torch.no_grad():
		for rgb_batch, targets, paths in tqdm(loader):
			if args.limit is not None and total_seen >= args.limit:
				break

			if args.limit is not None:
				remaining = args.limit - total_seen
				if remaining <= 0:
					break
				rgb_batch = rgb_batch[:remaining]
				targets = targets[:remaining]
				paths = paths[:remaining]

			rgb_batch = rgb_batch.to(device, non_blocking=True)
			targets = targets.to(device, non_blocking=True)
			logits = model(transform_cls.inverse_transform(rgb_batch))
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
							"original_class_name": category_name(target),
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
				"class_name": category_name(class_index),
				"correct": per_class_correct[class_index],
				"total": class_total,
				"accuracy": per_class_correct[class_index] / class_total,
			}
		)

	samples_output_path = output_dir / f"sample_attacks_{args.model_name}.json"
	performance_output_path = output_dir / f"val_performance_{args.model_name}.json"
	samples_output_path.write_text(json.dumps(correct_samples, indent=2), encoding="utf-8")
	performance_output_path.write_text(
		json.dumps(
			{
				"model_name": args.model_name,
				"checkpoint": str(resolved_checkpoint_path),
				"checkpoint_dir": str(checkpoint_dir),
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