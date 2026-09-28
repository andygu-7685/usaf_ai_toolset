"""Display grayscale image gradients in a Matplotlib window.

Run:
    python grayscale_gradient_viewer.py [optional_image_path]

If no image path is given, a file picker is shown.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from tkinter import Tk, filedialog

import cv2
import matplotlib.pyplot as plt
import numpy as np


IMAGE_FILE_TYPES = (
    ("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"),
    ("All files", "*.*"),
)


def pick_image_file() -> Path | None:
    """Open a native file picker and return the selected image path."""
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    try:
        filename = filedialog.askopenfilename(
            title="Choose a grayscale image",
            filetypes=IMAGE_FILE_TYPES,
        )
    finally:
        root.destroy()

    return Path(filename) if filename else None


def load_grayscale(image_path: Path) -> np.ndarray:
    """Load an image as a single-channel grayscale array."""
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise OSError(f"Could not open image: {image_path}")
    return image


def compute_gradients(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Sobel x/y gradients and gradient magnitude."""
    image_f32 = image.astype(np.float32)
    grad_x = cv2.Sobel(image_f32, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image_f32, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(grad_x, grad_y)
    return grad_x, grad_y, magnitude


def show_gradient_view(image: np.ndarray, grad_x: np.ndarray, grad_y: np.ndarray, magnitude: np.ndarray, image_path: Path) -> None:
    """Render original image and gradient maps in one Matplotlib figure."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle(f"Grayscale Gradients: {image_path.name}")

    ax0, ax1, ax2, ax3 = axes.ravel()

    ax0.imshow(image, cmap="gray", vmin=0, vmax=255)
    ax0.set_title("Original Grayscale")
    ax0.axis("off")

    # Center diverging maps around 0 to make positive/negative edges clearer.
    gx_lim = float(np.max(np.abs(grad_x))) or 1.0
    im1 = ax1.imshow(grad_x, cmap="seismic", vmin=-gx_lim, vmax=gx_lim)
    ax1.set_title("Gradient X (Sobel)")
    ax1.axis("off")
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    gy_lim = float(np.max(np.abs(grad_y))) or 1.0
    im2 = ax2.imshow(grad_y, cmap="seismic", vmin=-gy_lim, vmax=gy_lim)
    ax2.set_title("Gradient Y (Sobel)")
    ax2.axis("off")
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    im3 = ax3.imshow(magnitude, cmap="inferno")
    ax3.set_title("Gradient Magnitude")
    ax3.axis("off")
    fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    fig.tight_layout()
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize grayscale image gradients in Matplotlib.")
    parser.add_argument("image", nargs="?", help="Path to image file (optional).")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    image_path = Path(args.image) if args.image else pick_image_file()
    if image_path is None:
        print("No image selected.")
        return 0

    try:
        image = load_grayscale(image_path)
    except OSError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    grad_x, grad_y, magnitude = compute_gradients(image)
    show_gradient_view(image, grad_x, grad_y, magnitude, image_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
