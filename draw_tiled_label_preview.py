import argparse
from pathlib import Path
from PIL import Image, ImageDraw


def parse_yolo_pose_line(line: str):
    parts = line.strip().split()
    if len(parts) < 5:
        raise ValueError(f"Invalid label line: {line}")
    class_id = int(parts[0])
    x_center = float(parts[1])
    y_center = float(parts[2])
    width = float(parts[3])
    height = float(parts[4])
    keypoints = []
    for i in range(5, len(parts), 3):
        if i + 2 >= len(parts):
            break
        kp_x = float(parts[i])
        kp_y = float(parts[i + 1])
        kp_v = int(float(parts[i + 2]))
        keypoints.append((kp_x, kp_y, kp_v))
    return {
        "class_id": class_id,
        "x_center": x_center,
        "y_center": y_center,
        "width": width,
        "height": height,
        "keypoints": keypoints,
    }


def draw_labels_on_image(image_path: Path, label_path: Path, output_path: Path):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    img_w, img_h = image.size

    if not label_path.exists():
        raise FileNotFoundError(f"Label file not found: {label_path}")

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            label = parse_yolo_pose_line(line)
            x_center = label["x_center"] * img_w
            y_center = label["y_center"] * img_h
            width = label["width"] * img_w
            height = label["height"] * img_h
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2

            draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)

            visible_points = []
            for kp_x, kp_y, kp_v in label["keypoints"]:
                px = kp_x * img_w
                py = kp_y * img_h
                color = (0, 255, 0) if kp_v != 0 else (255, 255, 0)
                draw.ellipse([px - 3, py - 3, px + 3, py + 3], fill=color)
                if kp_v != 0:
                    visible_points.append((px, py))

            if len(visible_points) > 1:
                draw.line(visible_points, fill=(0, 255, 0), width=2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def process_dataset(image_dir: Path, label_dir: Path, preview_dir: Path):
    image_dir = image_dir.resolve()
    label_dir = label_dir.resolve()
    preview_dir = preview_dir.resolve()
    preview_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.png"))
    processed = 0

    for image_path in image_paths:
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            print(f"Skipping {image_path.name}: missing label file")
            continue

        preview_path = preview_dir / image_path.name
        draw_labels_on_image(image_path, label_path, preview_path)
        processed += 1

    print(f"Processed {processed} preview images into {preview_dir}")
    return processed


def main():
    parser = argparse.ArgumentParser(description="Draw YOLOv8 pose labels on tiled dataset images")
    parser.add_argument("--image_dir", required=True, help="Path to tiled images directory")
    parser.add_argument("--label_dir", required=True, help="Path to tiled labels directory")
    parser.add_argument("--preview_dir", required=False, default="preview", help="Output directory for preview images")
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    label_dir = Path(args.label_dir)
    preview_dir = Path(args.preview_dir)

    if not image_dir.exists() or not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not label_dir.exists() or not label_dir.is_dir():
        raise FileNotFoundError(f"Label directory not found: {label_dir}")

    process_dataset(image_dir, label_dir, preview_dir)


if __name__ == "__main__":
    main()
