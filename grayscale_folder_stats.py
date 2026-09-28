"""Pick a folder and print the darkest and brightest grayscale pixel value for each image.

Run:
    python grayscale_folder_stats.py

A native folder picker opens first. The script scans supported image files in the
chosen folder and prints each image's darkest and brightest grayscale pixel value.
"""

from __future__ import annotations

import sys
from pathlib import Path
from tkinter import Tk, filedialog

import numpy as np
from PIL import Image


SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".gif",
    ".tif",
    ".tiff",
    ".webp",
}


def pick_folder() -> Path | None:
    """Open a native folder picker and return the selected folder path."""
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    try:
        folder_name = filedialog.askdirectory(title="Choose a folder of images")
    finally:
        root.destroy()

    return Path(folder_name) if folder_name else None


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS


def grayscale_min_max(image_path: Path) -> tuple[int, int]:
    """Return darkest and brightest grayscale pixel values for an image."""
    image = Image.open(image_path).convert("L")
    grayscale = np.asarray(image, dtype=np.uint8)
    return int(grayscale.min()), int(grayscale.max())


def main() -> int:
    folder = pick_folder()
    if folder is None:
        print("No folder selected.")
        return 0

    if not folder.exists() or not folder.is_dir():
        print(f"Folder not found or not a directory: {folder}", file=sys.stderr)
        return 1

    image_files = sorted(path for path in folder.iterdir() if is_image_file(path))
    if not image_files:
        print("No supported image files found in the selected folder.")
        return 0

    for image_path in image_files:
        try:
            darkest, brightest = grayscale_min_max(image_path)
        except OSError as exc:
            print(f"{image_path.name}: could not open image ({exc})")
            continue

        print(f"{image_path.name}")
        print(f"  darkest: {darkest}")
        print(f"  brightest: {brightest}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
