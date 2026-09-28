import os
import random
import math
from PIL import Image, ImageDraw, ImageFont, ImageOps
import numpy as np
import cv2


IMG_SIZE = 1920
OUTPUT_DIR = "number_dataset"
IMAGE_DIR = os.path.join(OUTPUT_DIR, "images")
LABEL_DIR = os.path.join(OUTPUT_DIR, "labels")
PREVIEW_DIR = os.path.join(OUTPUT_DIR, "previews")

os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(LABEL_DIR, exist_ok=True)
os.makedirs(PREVIEW_DIR, exist_ok=True)


def disk_kernel(radius):
    kernel = np.zeros((2*radius+1, 2*radius+1), np.float32)
    # Draw a filled white circle on a black background
    cv2.circle(kernel, (radius, radius), radius, 1, -1)
    # Normalize so the sum is 1
    return kernel / (kernel.sum() * random.uniform(0.9, 2.0))



def load_font(size):
    """Try to load a truetype font, fallback to PIL default."""
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def create_number_image(number_text, font_size, angle, flip_horizontal, flip_vertical):
    """Create a rotated/flipped white number on a transparent canvas."""
    font = load_font(font_size)
    dummy = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
    draw = ImageDraw.Draw(dummy)
    text_bbox = draw.textbbox((0, 0), number_text, font=font)
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]

    padding = max(8, int(font_size * 0.15))
    canvas_w = text_w + padding * 2
    canvas_h = text_h + padding * 2
    temp_img = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(temp_img)
    draw.text((padding - text_bbox[0], padding - text_bbox[1]), number_text, fill=(255, 255, 255, 255), font=font)

    if flip_horizontal:
        temp_img = ImageOps.mirror(temp_img)
    if flip_vertical:
        temp_img = ImageOps.flip(temp_img)

    rotated = temp_img.rotate(angle, expand=True)

    if rotated.width > IMG_SIZE or rotated.height > IMG_SIZE:
        scale = min(IMG_SIZE / rotated.width, IMG_SIZE / rotated.height)
        new_size = (max(1, int(rotated.width * scale)), max(1, int(rotated.height * scale)))
        rotated = rotated.resize(new_size, resample=Image.LANCZOS)

    return rotated





def paste_shape(base_img, shape_img, existing_bboxes=None, max_attempts=50):
    """Paste the shape image onto a black base, avoiding overlap with existing boxes."""
    if existing_bboxes is None:
        existing_bboxes = []

    bg = base_img.copy()
    shape_w, shape_h = shape_img.size
    max_x = IMG_SIZE - shape_w
    max_y = IMG_SIZE - shape_h
    if max_x < 0 or max_y < 0:
        return None, None

    alpha = np.array(shape_img.split()[-1])
    nonzero = np.argwhere(alpha > 0)
    if nonzero.size == 0:
        return None, None

    y_min, x_min = nonzero.min(axis=0)
    y_max, x_max = nonzero.max(axis=0)
    shape_bbox = (x_min, y_min, x_max, y_max)

    for _ in range(max_attempts):
        paste_x = random.randint(0, max_x)
        paste_y = random.randint(0, max_y)
        candidate_bbox = (paste_x + shape_bbox[0], paste_y + shape_bbox[1], paste_x + shape_bbox[2], paste_y + shape_bbox[3])

        overlap = False
        for bx_min, by_min, bx_max, by_max in existing_bboxes:
            if not (candidate_bbox[2] < bx_min or candidate_bbox[0] > bx_max or candidate_bbox[3] < by_min or candidate_bbox[1] > by_max):
                overlap = True
                break
        if overlap:
            continue

        bg.paste(shape_img, (paste_x, paste_y), shape_img)
        return bg, candidate_bbox

    return None, None


def generate_number_pattern(index, output_count=1):
    """Generate one image with multiple white numbers on black background and save bbox file."""
    object_count = 30
    base_img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (0, 0, 0))
    current_img = base_img.copy()
    bboxes = []
    labels = []

    for _ in range(object_count):
        number_text = str(random.randint(0, 9))
        if len(number_text) > 3:
            number_text = number_text[:3]

        font_size = random.randint(20, 150)
        angle = random.uniform(-90, 90)
        flip_horizontal = random.choice([False, True])
        flip_vertical = random.choice([False, True])

        shape_img = create_number_image(number_text, font_size, angle, flip_horizontal, flip_vertical)
        if shape_img.width > IMG_SIZE or shape_img.height > IMG_SIZE:
            scale = min(IMG_SIZE / shape_img.width, IMG_SIZE / shape_img.height)
            shape_img = shape_img.resize((max(1, int(shape_img.width * scale)), max(1, int(shape_img.height * scale))), resample=Image.LANCZOS)

        pasted = paste_shape(current_img, shape_img, existing_bboxes=bboxes)
        if pasted[0] is None:
            continue

        current_img, bbox = pasted
        bboxes.append(bbox)
        labels.append((number_text, bbox))

    if not labels:
        raise RuntimeError("Failed to place any number in the image.")

    annotation_img = current_img.copy()

    for number_text, (x_min, y_min, x_max, y_max) in labels:
        draw = ImageDraw.Draw(annotation_img)
        draw.rectangle((x_min, y_min, x_max, y_max), outline="red", width=2)

    # 1. Create your 9x9 kernel
    kernel = disk_kernel(4) 

    # 2. Convert your Pillow image to a NumPy array (OpenCV format)
    img_array = np.array(current_img)

    # 3. Apply the filter using OpenCV
    # -1 means the output image will have the same depth as the input
    filtered_img_array = cv2.filter2D(img_array, -1, kernel)
    current_img = Image.fromarray(filtered_img_array)

    image_path = os.path.join(IMAGE_DIR, f"number_pattern_{index:04d}.jpg")
    label_path = os.path.join(LABEL_DIR, f"number_pattern_{index:04d}.txt")
    current_img.save(image_path)
    annotation_path = os.path.join(PREVIEW_DIR, f"number_pattern_{index:04d}_annotation.jpg")
    annotation_img.save(annotation_path)

    with open(label_path, "w", encoding="utf-8") as f:
        for number_text, (x_min, y_min, x_max, y_max) in labels:
            f.write(f"{0} {x_min} {y_min} {x_max} {y_max}\n")
            

    return image_path, label_path, bboxes


def generate_dataset(count=100):
    """Generate a dataset of number images and label files."""
    for i in range(count):
        image_path, label_path, bbox = generate_number_pattern(i)
        print(f"Saved {image_path} and {label_path} bbox={bbox}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate black images with rotated/flipped white numbers and bounding boxes.")
    parser.add_argument("--count", type=int, default=100, help="Number of images to generate")
    parser.add_argument("--size", type=int, default=IMG_SIZE, help="Image width and height")
    args = parser.parse_args()

    IMG_SIZE = args.size
    generate_dataset(args.count)
