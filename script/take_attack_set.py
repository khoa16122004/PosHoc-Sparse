from __future__ import annotations

import argparse
import json
from pathlib import Path



def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Select one preserved sample per class from an evaluator JSON file."
	)
	parser.add_argument(
		"--input-json",
		type=Path,
        required=True,
		help="Path to the evaluator sample JSON.",
	)
	parser.add_argument(
		"--output-json",
		type=Path,
		default=None,
		help="Optional output path. Defaults to selected_1000.json next to the input file.",
	)
	return parser.parse_args()


def select_one_sample_per_class(samples: list[dict]) -> list[dict]:
	selected_by_class: dict[int, dict] = {}

	for sample in samples:
		original_class = int(sample["original_class"])
		if original_class not in selected_by_class:
			selected_by_class[original_class] = sample

	return [selected_by_class[class_id] for class_id in sorted(selected_by_class)]


def main(args: argparse.Namespace) -> None:
	input_json = args.input_json
	output_json = args.output_json or input_json.with_name("selected_1000.json")

	with input_json.open("r", encoding="utf-8") as file_obj:
		samples = json.load(file_obj)

	if not isinstance(samples, list):
		raise ValueError("Input JSON must be a list of sample records.")

	selected_samples = select_one_sample_per_class(samples)
	output_json.write_text(json.dumps(selected_samples, indent=2), encoding="utf-8")

	print(f"Input samples: {len(samples)}")
	print(f"Selected samples: {len(selected_samples)}")
	print(f"Saved selected samples to: {output_json}")


if __name__ == "__main__":
	main(parse_args())
