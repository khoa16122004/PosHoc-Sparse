
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

try:
	from PIL import Image
except ImportError:
	Image = None


DEFAULT_INPUT_JSON = Path(
	r"D:\PosHoc-Sparse\evaluate_results\SIGLIP\google\siglip-base-patch16-224\selected_1000.json"
)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"Create random 2x2 image grids from a selected_1000.json-style file. "
			"Each output grid contains 4 sampled source records."
		)
	)
	parser.add_argument(
		"--input-json",
		type=Path,
		default=DEFAULT_INPUT_JSON,
		help="Path to the selected samples JSON file.",
	)
	parser.add_argument(
		"--output-json",
		type=Path,
		default=None,
		help="Output path for the generated grid manifest. Defaults next to the input file.",
	)
	parser.add_argument(
		"--grid-count",
		type=int,
		default=500,
		help="Number of random grids to create.",
	)
	parser.add_argument(
		"--grid-size",
		type=int,
		default=4,
		help="Number of images per grid. The default 4 produces a 2x2 layout.",
	)
	parser.add_argument(
		"--seed",
		type=int,
		default=0,
		help="Random seed for reproducible sampling.",
	)
	parser.add_argument(
		"--save-images",
		action="store_true",
		help="Also save composite grid images. Requires Pillow and valid source image paths.",
	)
	parser.add_argument(
		"--image-output-dir",
		type=Path,
		default=None,
		help="Directory for composite images. Defaults to a random_grid_images folder next to the output JSON.",
	)
	parser.add_argument(
		"--image-size",
		type=int,
		default=224,
		help="Size in pixels for each cell inside a saved composite image.",
	)
	return parser.parse_args()


def load_samples(input_json: Path) -> list[dict[str, Any]]:
	with input_json.open("r", encoding="utf-8") as file_obj:
		samples = json.load(file_obj)

	if not isinstance(samples, list):
		raise ValueError("Input JSON must be a list of sample records.")

	if not samples:
		raise ValueError("Input JSON is empty.")

	return samples


def build_grid_manifest(
	samples: list[dict[str, Any]],
	grid_count: int,
	grid_size: int,
	seed: int,
) -> list[dict[str, Any]]:
	if grid_count <= 0:
		raise ValueError("grid_count must be greater than 0.")

	if grid_size <= 0:
		raise ValueError("grid_size must be greater than 0.")

	if len(samples) < grid_size:
		raise ValueError(
			f"Need at least {grid_size} samples, but only found {len(samples)}."
		)

	rng = random.Random(seed)
	grids: list[dict[str, Any]] = []

	for grid_id in range(grid_count):
		selected_samples = rng.sample(samples, k=grid_size)
		grids.append(
			{
				"grid_id": grid_id,
				"component_count": grid_size,
				"components": selected_samples,
			}
		)

	return grids


def save_grid_images(
	grids: list[dict[str, Any]],
	output_dir: Path,
	image_size: int,
) -> None:
	if Image is None:
		raise ImportError("Pillow is required for --save-images. Install it with `pip install pillow`.")

	if image_size <= 0:
		raise ValueError("image_size must be greater than 0.")

	output_dir.mkdir(parents=True, exist_ok=True)

	for grid in grids:
		components = grid["components"]
		if len(components) != 4:
			raise ValueError("Composite image export currently supports grid-size=4 only.")

		canvas = Image.new("RGB", (image_size * 2, image_size * 2), color=(0, 0, 0))
		positions = [(0, 0), (image_size, 0), (0, image_size), (image_size, image_size)]

		for component, position in zip(components, positions):
			img_path = component.get("img_path")
			if not img_path:
				raise ValueError("Each component must include an img_path to save composite images.")

			source_path = Path(img_path)
			if not source_path.exists():
				raise FileNotFoundError(f"Source image not found: {source_path}")

			with Image.open(source_path) as source_image:
				tile = source_image.convert("RGB").resize((image_size, image_size))
				canvas.paste(tile, position)

		output_path = output_dir / f"grid_{grid['grid_id']:04d}.jpg"
		canvas.save(output_path, quality=95)


def main(args: argparse.Namespace) -> None:
	input_json = args.input_json
	output_json = args.output_json or input_json.with_name("random_grids.json")
	image_output_dir = args.image_output_dir or output_json.with_name("random_grid_images")

	samples = load_samples(input_json)
	grids = build_grid_manifest(
		samples=samples,
		grid_count=args.grid_count,
		grid_size=args.grid_size,
		seed=args.seed,
	)

	payload = {
		"source_json": str(input_json),
		"grid_count": args.grid_count,
		"grid_size": args.grid_size,
		"seed": args.seed,
		"grids": grids,
	}
	output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

	print(f"Loaded samples: {len(samples)}")
	print(f"Saved grid manifest to: {output_json}")

	if args.save_images:
		save_grid_images(grids=grids, output_dir=image_output_dir, image_size=args.image_size)
		print(f"Saved composite images to: {image_output_dir}")


if __name__ == "__main__":
	main(parse_args())