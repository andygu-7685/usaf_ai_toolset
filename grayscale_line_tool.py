from __future__ import annotations
import sys
from pathlib import Path
import shutil
from tkinter import Tk, filedialog
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button
from PIL import Image

SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
NULL_FOLDER_NAME = "null"
BUCKET_COUNT = 10
CLASSIFIED_IMAGES_FOLDER_NAME = "classified_scanlines"
CLASSIFIED_LABELS_FOLDER_NAME = "classified_scanlines_labels"
DEBUG = False


def pick_image_folder() -> Path | None:
    """Open a native folder picker and return the selected directory."""
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    try:
        foldername = filedialog.askdirectory(
            title="Choose an image folder",
        )
    finally:
        root.destroy()

    return Path(foldername) if foldername else None


def list_image_files(folder_path: Path) -> list[Path]:
    """Return supported images in the selected folder."""
    return sorted(
        path for path in folder_path.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )


def _create_bucket_folders(output_root: Path) -> dict[str, Path]:
    """Create bucket folders for all contrast ranges and null."""
    output_root.mkdir(parents=True, exist_ok=True)

    output_dirs: dict[str, Path] = {}
    for bucket_index in range(BUCKET_COUNT):
        start = bucket_index / BUCKET_COUNT
        end = (bucket_index + 1) / BUCKET_COUNT
        folder_name = f"{start:.1f}-{end:.1f}"
        folder_path = output_root / folder_name
        folder_path.mkdir(exist_ok=True)
        output_dirs[folder_name] = folder_path

    null_path = output_root / NULL_FOLDER_NAME
    null_path.mkdir(exist_ok=True)
    output_dirs[NULL_FOLDER_NAME] = null_path
    return output_dirs


def create_output_folders(source_folder: Path) -> dict[str, Path]:
    """Create image output folders beside the source folder for each contrast bucket."""
    return _create_bucket_folders(source_folder / CLASSIFIED_IMAGES_FOLDER_NAME)


def create_label_folders(source_folder: Path) -> dict[str, Path]:
    """Create label output folders mirroring image bucket folders."""
    return _create_bucket_folders(source_folder / CLASSIFIED_LABELS_FOLDER_NAME)


def load_processed_grayscale(image_path: Path) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Load grayscale and return display and measurement images."""
    image = Image.open(image_path).convert("L")
    grayscale = np.asarray(image, dtype=np.float32)

    darkest = float(grayscale.min())
    brightest = float(grayscale.max())
    value_range = brightest - darkest

    if value_range == 0:
        display_image = np.zeros_like(grayscale, dtype=np.float32)
    else:
        display_image = (grayscale - darkest) / value_range

    measurement_image = display_image.copy()
    return display_image, measurement_image, darkest, brightest


def pixel_values_along_line(image: np.ndarray, start: tuple, end: tuple) -> np.ndarray:
    """Sample pixel values along a line using linear interpolation."""
    # Calculate the number of steps needed to cover the distance
    dist = np.linalg.norm(np.array(end) - np.array(start))
    n_samples = int(np.ceil(dist)) + 1
    
    # Generate linear coordinate arrays
    x = np.linspace(start[0], end[0], n_samples).round().astype(int)
    y = np.linspace(start[1], end[1], n_samples).round().astype(int)

    if DEBUG:
        print(f"Distance: {dist}")
        print(f"Number of samples: {n_samples}")
        print(f"X coords: {x}")
        print(f"Y coords: {y}")
    
    # Constrain coordinates to image boundaries
    x = np.clip(x, 0, image.shape[1] - 1)
    y = np.clip(y, 0, image.shape[0] - 1)
    
    return image[y, x]


def contrast_bucket_name(difference: float) -> str:
    """Map a normalized grayscale difference into a 10% bucket."""
    clipped = min(max(difference, 0.0), 1.0)
    bucket_index = min(int(clipped * BUCKET_COUNT), BUCKET_COUNT - 1)
    start = bucket_index / BUCKET_COUNT
    end = (bucket_index + 1) / BUCKET_COUNT
    if DEBUG:
        print(f"Difference {difference:.4f} mapped to bucket {bucket_index} ({start:.1f}-{end:.1f})")
    return f"{start:.1f}-{end:.1f}"


class FolderMeasurementViewer:
    """Interactive Matplotlib viewer for classifying images by line contrast."""

    def __init__(self, image_paths: list[Path], output_dirs: dict[str, Path], label_dirs: dict[str, Path]):
        self.image_paths = image_paths
        self.output_dirs = output_dirs
        self.label_dirs = label_dirs
        self.current_index = 0
        self.is_complete = False
        self.display_image: np.ndarray | None = None
        self.measurement_image: np.ndarray | None = None
        self.image_path: Path | None = None
        self.source_min = 0.0
        self.source_max = 0.0
        self.start: tuple[float, float] | None = None
        self.line_artist = None

        self.figure, self.axes = plt.subplots()
        self.figure.subplots_adjust(bottom=0.01, top=0.70)
        self.image_artist = self.axes.imshow(
            np.zeros((1, 1), dtype=np.float32),
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
            origin="upper",
        )
        self.axes.format_coord = lambda x, y: f"x={int(round(x))}, y={int(round(y))}"
        self.axes.set_axis_off()

        self.status_text = self.figure.suptitle(
            self._default_status(),
            x=0.02,
            y=0.98,
            ha="left",
            va="top",
            fontsize=9,
        )

        self.null_button = Button(self.figure.add_axes([0.78, 0.05, 0.16, 0.07]), "Mark Null")
        self.null_button.on_clicked(self.on_null_clicked)

        self.figure.canvas.mpl_connect("button_press_event", self.on_press)
        self.figure.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.figure.canvas.mpl_connect("button_release_event", self.on_release)
        self._load_current_image()

    def _default_status(self) -> str:
        image_name = self.image_path.name if self.image_path is not None else "No image"
        return (
            f"{image_name}\n"
            f"Image {self.current_index + 1}/{len(self.image_paths)}\n"
            "Draw a line with the left mouse button\n"
            f"Grayscale min: {self.source_min:.0f}, max: {self.source_max:.0f}"
        )

    def _reset_line(self) -> None:
        if self.line_artist is not None:
            self.line_artist.remove()
            self.line_artist = None

    def _load_current_image(self) -> None:
        self.start = None
        self._reset_line()

        self.image_path = self.image_paths[self.current_index]
        self.display_image, self.measurement_image, self.source_min, self.source_max = load_processed_grayscale(
            self.image_path
        )
        self.image_artist.set_data(self.display_image)
        height, width = self.display_image.shape
        self.image_artist.set_extent((-0.5, width - 0.5, height - 0.5, -0.5))
        self.axes.set_xlim(-0.5, width - 0.5)
        self.axes.set_ylim(height - 0.5, -0.5)
        self.axes.set_title(f"{self.image_path.name} - draw to classify")
        self.status_text.set_text(self._default_status())
        self.figure.canvas.draw_idle()

    def _copy_current_image(self, destination_dir: Path) -> Path | None:
        if self.image_path is None:
            return

        destination = destination_dir / self.image_path.name
        counter = 1
        while destination.exists():
            destination = destination_dir / f"{self.image_path.stem}_{counter}{self.image_path.suffix}"
            counter += 1

        shutil.copy2(self.image_path, destination)
        return destination

    def _write_yolov8_scanline_label(
        self,
        label_path: Path,
        start: tuple[float, float],
        end: tuple[float, float],
    ) -> None:
        if self.measurement_image is None:
            return

        height, width = self.measurement_image.shape
        if width <= 0 or height <= 0:
            return

        x1 = float(np.clip(start[0], 0.0, width - 1.0))
        y1 = float(np.clip(start[1], 0.0, height - 1.0))
        x2 = float(np.clip(end[0], 0.0, width - 1.0))
        y2 = float(np.clip(end[1], 0.0, height - 1.0))

        xmin, xmax = min(x1, x2), max(x1, x2)
        ymin, ymax = min(y1, y2), max(y1, y2)

        x_center = ((xmin + xmax) * 0.5) / width
        y_center = ((ymin + ymax) * 0.5) / height
        box_width = (xmax - xmin) / width
        box_height = (ymax - ymin) / height

        x1n = x1 / width
        y1n = y1 / height
        x2n = x2 / width
        y2n = y2 / height

        # YOLOv8 pose line: class cx cy w h x1 y1 v1 x2 y2 v2
        line = (
            f"0 {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f} "
            f"{x1n:.6f} {y1n:.6f} 2 {x2n:.6f} {y2n:.6f} 2\n"
        )
        swapped_line = (
            f"0 {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f} "
            f"{x2n:.6f} {y2n:.6f} 2 {x1n:.6f} {y1n:.6f} 2\n"
        )
        label_path.write_text(line + swapped_line, encoding="utf-8")

    def _write_empty_label(self, label_path: Path) -> None:
        label_path.write_text("", encoding="utf-8")

    def _advance_to_next_image(self, message: str, path: Path | None = None) -> None:
        if self.current_index >= len(self.image_paths) - 1:
            self.is_complete = True
            if path is not None:
                self.status_text.set_text(f"{path}\n{message}\nFinished all images.")
            else:
                self.status_text.set_text(f"{message}\nFinished all images.")
            self.axes.set_title("Classification complete")
            self._reset_line()
            self.figure.canvas.draw_idle()
            return

        self.current_index += 1
        self._load_current_image()
        if path is not None:
            self.status_text.set_text(f"{path}\n{message}\n\n{self._default_status()}")
        else:
            self.status_text.set_text(f"{message}\n\n{self._default_status()}")
        self.figure.canvas.draw_idle()

    def _classify_current_image(
        self,
        folder_name: str,
        message: str,
        start: tuple[float, float] | None = None,
        end: tuple[float, float] | None = None,
    ) -> None:
        path = self._copy_current_image(self.output_dirs[folder_name])
        if path is not None:
            label_path = self.label_dirs[folder_name] / f"{path.stem}.txt"
            if start is not None and end is not None:
                self._write_yolov8_scanline_label(label_path, start, end)
            else:
                self._write_empty_label(label_path)
        self._advance_to_next_image(message, path)

    def on_press(self, event) -> None:
        if self.is_complete:
            return
        if event.inaxes != self.axes or event.button != 1:
            return
        if event.xdata is None or event.ydata is None:
            return

        self.start = (event.xdata, event.ydata)
        self._reset_line()

        (self.line_artist,) = self.axes.plot(
            [event.xdata, event.xdata],
            [event.ydata, event.ydata],
            color="red",
            linewidth=2,
        )
        self.figure.canvas.draw_idle()

    def on_motion(self, event) -> None:
        if self.is_complete:
            return
        if self.start is None or self.line_artist is None:
            return
        if event.inaxes != self.axes or event.xdata is None or event.ydata is None:
            return

        self.line_artist.set_data([self.start[0], event.xdata], [self.start[1], event.ydata])
        self.figure.canvas.draw_idle()

    def on_release(self, event) -> None:
        if self.is_complete:
            return
        if self.start is None or self.line_artist is None or self.measurement_image is None:
            return
        if event.inaxes != self.axes or event.xdata is None or event.ydata is None:
            self.start = None
            self._reset_line()
            self.figure.canvas.draw_idle()
            return

        end = (event.xdata, event.ydata)
        self.line_artist.set_data([self.start[0], end[0]], [self.start[1], end[1]])

        values = pixel_values_along_line(self.measurement_image, self.start, end)
        if DEBUG:
            print(f"pixel values along line: {values}")
            print("start:", self.start, "end:", end)
        darkest = float(values.min())
        brightest = float(values.max())
        difference = brightest - darkest
        bucket_name = contrast_bucket_name(difference)

        self._classify_current_image(
            bucket_name,
            "Normalized grayscale values along line\n"
            f"Darkest: {darkest:.4f}\n"
            f"Brightest: {brightest:.4f}\n"
            f"Difference: {difference:.4f}\n"
            f"Saved to: {bucket_name}\n"
            f"Sampled pixels: {len(values)}",
            self.start,
            end,
        )
        self.start = None

    def on_null_clicked(self, _event) -> None:
        if self.is_complete:
            return
        self._classify_current_image(
            NULL_FOLDER_NAME,
            f"Marked as {NULL_FOLDER_NAME} and saved to {NULL_FOLDER_NAME}.",
        )

    def show(self) -> None:
        plt.show()


def main() -> int:
    folder_path = pick_image_folder()
    if folder_path is None:
        print("No folder selected.")
        return 0

    image_paths = list_image_files(folder_path)
    if not image_paths:
        print("No supported images found in the selected folder.", file=sys.stderr)
        return 1

    output_dirs = create_output_folders(folder_path)
    label_dirs = create_label_folders(folder_path)

    try:
        viewer = FolderMeasurementViewer(image_paths, output_dirs, label_dirs)
    except OSError as exc:
        print(f"Could not open image: {exc}", file=sys.stderr)
        return 1

    viewer.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
