"""
Prepare Stanford Cars data from Hugging Face for the local training scripts.

The training scripts in this project expect the original Stanford Cars layout:

    data/
      cars_train/
      devkit/
        cars_train_annos.mat
        cars_meta.mat

The original download links are no longer reliable, so this script converts the
Hugging Face mirror into that layout.
"""
import argparse
from pathlib import Path

import numpy as np
import scipy.io as sio
from datasets import load_dataset
from tqdm import tqdm


def save_meta(class_names: list[str], devkit_dir: Path) -> None:
    meta = np.empty((1, len(class_names)), dtype=object)
    for idx, name in enumerate(class_names):
        meta[0, idx] = name
    sio.savemat(devkit_dir / "cars_meta.mat", {"class_names": meta})


def save_annotations(records: list[tuple[str, int, int, int]], devkit_dir: Path) -> None:
    dtype = [
        ("bbox_x1", "O"),
        ("bbox_y1", "O"),
        ("bbox_x2", "O"),
        ("bbox_y2", "O"),
        ("class", "O"),
        ("fname", "O"),
    ]
    annotations = np.empty((1, len(records)), dtype=dtype)

    for idx, (fname, label, width, height) in enumerate(records):
        annotations[0, idx]["bbox_x1"] = np.array([[1]])
        annotations[0, idx]["bbox_y1"] = np.array([[1]])
        annotations[0, idx]["bbox_x2"] = np.array([[width]])
        annotations[0, idx]["bbox_y2"] = np.array([[height]])
        annotations[0, idx]["class"] = np.array([[label + 1]])
        annotations[0, idx]["fname"] = fname

    sio.savemat(devkit_dir / "cars_train_annos.mat", {"annotations": annotations})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="data", help="Output directory under backend/")
    parser.add_argument("--dataset", default="tanganke/stanford_cars")
    parser.add_argument("--split", default="train")
    parser.add_argument("--max_samples", type=int, default=0, help="Optional small smoke-test limit")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    image_dir = output_dir / "cars_train"
    devkit_dir = output_dir / "devkit"
    image_dir.mkdir(parents=True, exist_ok=True)
    devkit_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(args.dataset, split=args.split, streaming=True)
    class_names = ds.features["label"].names
    save_meta(class_names, devkit_dir)

    records: list[tuple[str, int, int, int]] = []
    for idx, sample in enumerate(tqdm(ds, desc="Saving images"), start=1):
        if args.max_samples and idx > args.max_samples:
            break
        img = sample["image"].convert("RGB")
        label = int(sample["label"])
        fname = f"{idx:05d}.jpg"
        img.save(image_dir / fname, "JPEG", quality=95)
        records.append((fname, label, img.width, img.height))

    save_annotations(records, devkit_dir)
    print(f"Saved {len(records)} images to {image_dir}")
    print(f"Saved metadata to {devkit_dir}")


if __name__ == "__main__":
    main()
