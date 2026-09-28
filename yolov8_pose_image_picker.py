"""Pick an image, rotate it through multiple angles, and run YOLOv8 pose.

Run:
    python yolov8_pose_image_picker.py

The script opens a native image picker, rotates the image at predefined angles
with an expanded canvas (no cropping), runs pose inference for each rotated
image, and shows all angle results in one Matplotlib window.
"""

from __future__ import annotations

import sys
from pathlib import Path
from tkinter import Tk, filedialog

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


MODEL_FILE_TYPES = (
    ("PyTorch model", "*.pt"),
    ("All files", "*.*"),
)

IMAGE_FILE_TYPES = (
    ("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"),
    ("All files", "*.*"),
)

POSE_CONNECTIONS = (
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
)

KEYPOINT_CONFIDENCE_THRESHOLD = 0.25
ROTATION_ANGLES = (-180, -165, -150, -135, -120, -90, -75, -60, -45, -30, -15, 0, 15, 30, 45, 60, 75, 90, 120, 135, 150, 165, 180)


def pick_file(title: str, filetypes: tuple[tuple[str, str], ...]) -> Path | None:
    """Open a native file picker and return the selected path."""
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    try:
        filename = filedialog.askopenfilename(title=title, filetypes=filetypes)
    finally:
        root.destroy()

    return Path(filename) if filename else None


def rotate_image_expand_canvas(image: Image.Image, angle: float) -> Image.Image:
    """Rotate image without cropping by expanding the destination canvas."""
    return image.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)


def draw_pose_on_axes(axes, image_array: np.ndarray, result, title: str) -> None:
    """Draw detected keypoints and skeleton connections on one axes."""
    axes.imshow(image_array)
    axes.set_title(title)
    axes.set_axis_off()

    keypoints = result.keypoints
    if keypoints is None or keypoints.xy is None or len(keypoints.xy) == 0:
        return

    confidences = keypoints.conf.tolist() if keypoints.conf is not None else None
    colors = ("red", "lime", "cyan", "yellow", "orange", "magenta")

    for pose_index, points in enumerate(keypoints.xy.tolist()):
        color = colors[pose_index % len(colors)]
        point_confidences = confidences[pose_index] if confidences is not None else None

        for point_index, (x_coord, y_coord) in enumerate(points):
            if point_confidences is not None:
                confidence = point_confidences[point_index]
                if confidence < KEYPOINT_CONFIDENCE_THRESHOLD:
                    continue

            axes.scatter(x_coord, y_coord, color=color, s=28)

        for start_index, end_index in POSE_CONNECTIONS:
            if start_index >= len(points) or end_index >= len(points):
                continue

            if point_confidences is not None:
                if (
                    point_confidences[start_index] < KEYPOINT_CONFIDENCE_THRESHOLD
                    or point_confidences[end_index] < KEYPOINT_CONFIDENCE_THRESHOLD
                ):
                    continue

            x_start, y_start = points[start_index]
            x_end, y_end = points[end_index]
            axes.plot((x_start, x_end), (y_start, y_end), color=color, linewidth=2)


def show_pose_plots_for_angles(
    image_name: str,
    rotated_images: list[np.ndarray],
    results,
    angles: tuple[float, ...],
) -> None:
    """Show all rotated-image pose results in one Matplotlib window."""
    total = len(angles)
    columns = 3
    rows = (total + columns - 1) // columns

    figure, axes_grid = plt.subplots(rows, columns, figsize=(5 * columns, 4 * rows))
    axes_array = np.atleast_1d(axes_grid).ravel()

    for index, angle in enumerate(angles):
        draw_pose_on_axes(
            axes_array[index],
            rotated_images[index],
            results[index],
            f"{image_name} | {angle:+.0f} deg",
        )

    for index in range(total, len(axes_array)):
        axes_array[index].set_axis_off()

    figure.tight_layout()

    plt.show()


def main() -> int:
    if YOLO is None:
        print(
            "Ultralytics is not installed. Install it with: pip install ultralytics",
            file=sys.stderr,
        )
        return 1

    model_path = Path("model/single_pattern_scanline_v2.pt")

    image_path = pick_file("Choose an image", IMAGE_FILE_TYPES)
    if image_path is None:
        print("No image selected.")
        return 0

    if not model_path.exists() or not model_path.is_file():
        print(f"Model file not found: {model_path}", file=sys.stderr)
        return 1

    if not image_path.exists() or not image_path.is_file():
        print(f"Image file not found: {image_path}", file=sys.stderr)
        return 1

    try:
        model = YOLO(str(model_path))
    except Exception as exc:
        print(f"Could not load model: {exc}", file=sys.stderr)
        return 1

    try:
        original = Image.open(image_path).convert("RGB")
    except OSError as exc:
        print(f"Could not open image: {exc}", file=sys.stderr)
        return 1

    rotated_images: list[np.ndarray] = []
    per_angle_results = []

    for angle in ROTATION_ANGLES:
        rotated = rotate_image_expand_canvas(original, angle)
        rotated_array = np.asarray(rotated)
        rotated_images.append(rotated_array)

        try:
            angle_results = model(rotated_array)
        except Exception as exc:
            print(f"Could not run model at angle {angle:+.0f}: {exc}", file=sys.stderr)
            return 1

        if not angle_results:
            print(f"{angle:+.0f} deg: model returned no result.")
            return 1

        result = angle_results[0]
        per_angle_results.append(result)

        keypoints = result.keypoints
        if keypoints is None or keypoints.xy is None or len(keypoints.xy) == 0:
            print(f"{angle:+.0f} deg: no pose keypoints detected.")
            continue

        print(f"{angle:+.0f} deg: detected {len(keypoints.xy)} pose instance(s)")

    show_pose_plots_for_angles(
        image_path.name,
        rotated_images,
        per_angle_results,
        ROTATION_ANGLES,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())