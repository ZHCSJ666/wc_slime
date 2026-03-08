import argparse
import os
from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--hf-dataset",
        type=str,
        required=True,
        help="Hugging Face dataset name, e.g. ZHCSJ/wc-en-open-slime-4k",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Local output directory for the prepared parquet dataset",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print(f"Loading dataset from Hugging Face: {args.hf_dataset}")
    print("=" * 80)

    ds = load_dataset(args.hf_dataset, split="train")

    print("Dataset loaded successfully:")
    print(ds)

    if len(ds) == 0:
        raise ValueError("Downloaded dataset is empty.")

    sample = ds[0]
    print("First sample keys:", list(sample.keys()))

    required_keys = ["prompt", "label", "images"]
    for key in required_keys:
        if key not in sample:
            raise ValueError(f"Dataset missing required field: {key}")

    output_path = os.path.join(args.output_dir, "train.parquet")

    print(f"Saving dataset to parquet: {output_path}")
    ds.to_parquet(output_path)

    if not os.path.isfile(output_path):
        raise FileNotFoundError(f"Failed to create parquet file: {output_path}")

    print("=" * 80)
    print("Dataset preparation finished successfully.")
    print(f"Saved file: {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()