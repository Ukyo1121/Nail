#!/usr/bin/env python3
"""Add quality attribute to images in annotation JSON files.

val_classification_A.json -> quality = 0
val_classification_B.json -> quality = 1
"""

import json
import os

ANNOTATIONS_DIR = os.path.join(os.path.dirname(__file__), "..", "annotations")

FILES = {
    "val_classification_A.json": 0,
    "val_classification_B.json": 1,
}


def add_quality(filepath: str, quality: int) -> None:
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    for img in data.get("images", []):
        img["quality"] = quality

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print(f"Added quality={quality} to {len(data.get('images', []))} images in {os.path.basename(filepath)}")


def main():
    for filename, quality in FILES.items():
        filepath = os.path.join(ANNOTATIONS_DIR, filename)
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            continue
        add_quality(filepath, quality)


if __name__ == "__main__":
    main()
