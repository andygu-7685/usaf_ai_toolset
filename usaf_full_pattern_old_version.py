import os
import random
import math
from PIL import Image, ImageDraw, ImageOps
import numpy as np
from scipy import ndimage

# ---------- configuration ----------
IMG_SIZE = 1920
OUTPUT_DIR = "usaf_full_dataset"
IMAGE_DIR = os.path.join(OUTPUT_DIR, "images")
LABEL_DIR = os.path.join(OUTPUT_DIR, "labels")
PREVIEW_DIR = os.path.join(OUTPUT_DIR, "preview")

os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(LABEL_DIR, exist_ok=True)
os.makedirs(PREVIEW_DIR, exist_ok=True)

# ---------- helper: image operations ----------

def load_source(path: str) -> Image.Image:
    """Load an image file and return it as RGBA."""
    return Image.open(path).convert("RGBA")


def scale_to_canvas(img: Image.Image, canvas_size: int = IMG_SIZE) -> Image.Image:
    """
    Resize the image so its longer side occupies 60-80 % of the canvas.
    """
    target_pct = random.uniform(0.4, 0.9)
    target_px = int(canvas_size * target_pct)

    w, h = img.size
    if w >= h:
        new_w = target_px
        new_h = max(1, int(h * target_px / w))
    else:
        new_h = target_px
        new_w = max(1, int(w * target_px / h))

    return img.resize((new_w, new_h), Image.LANCZOS)


def apply_random_flips(img: Image.Image) -> Image.Image:
    """Randomly apply horizontal and/or vertical flips."""
    mirrored = False
    fliped = False
    if random.random() < 0.5:
        img = ImageOps.mirror(img)
        mirrored = True
    if random.random() < 0.5:
        img = ImageOps.flip(img)
        fliped = True
    return img, mirrored, fliped


def rotate_image(img: Image.Image, angle: float, mirror: bool = False, flip: bool = False):
    w, h = img.size
    # Convert angle to radians and handle CCW direction
    # Pillow rotates CCW, which matches the standard math direction
    rad = math.radians(angle)
    
    # 1. Rotate the image
    rotated = img.rotate(-angle, expand=True, resample=Image.BICUBIC)
    rw, rh = rotated.size
    
    # 2. Define original corners relative to the original center
    # Center of original image is (w/2, h/2)
    cx, cy = w / 2.0, h / 2.0
    original_corners = np.array([
        [-cx, -cy], # Top-left
        [ (w-cx), -cy], # Top-right
        [ (w-cx),  (h-cy)], # Bottom-right
        [-cx,  (h-cy)]  # Bottom-left
    ])

    # rearrange corners if mirrored or flipped to maintain consistent order (TL, TR, BR, BL)
    if mirror:
        # Swap left and right corners
        original_corners[[0, 1, 2, 3]] = original_corners[[1, 0, 3, 2]]
    if flip:
        # Swap top and bottom corners
        original_corners[[0, 1, 2, 3]] = original_corners[[3, 2, 1, 0]]

    shift_corners_1 = np.array([
        [0.83, 0.73], # Top-left
        [0.814, 0.729], # Top-right
        [0.824, 0.957], # Bottom-right
        [0.827, 0.96]  # Bottom-left
    ])
    original_corners *= shift_corners_1
    
    shift_corners_2 = np.array([
        [0.055, 0.154], # Top-left
        [0.45, 0.153], # Top-right
        [0.448, 0.325], # Bottom-right
        [0.053, 0.325]  # Bottom-left
    ])
    secondary_corners = original_corners * shift_corners_2
    
    # Calculate the center of the secondary corners
    secondary_center = np.mean(secondary_corners, axis=0)
    secondary_corners_centered = secondary_corners - secondary_center
    
    shift_corners_3 = np.array([
        [0.055, 0.264], # Top-left
        [0.45, 0.263], # Top-right
        [0.448, 0.235], # Bottom-right
        [0.053, 0.235]  # Bottom-left
    ])
    tertiary_corners = secondary_corners_centered * shift_corners_3
    # shift tertiary corners back to original center
    tertiary_corners += secondary_center
    
    # 3. Rotation Matrix for CCW rotation
    # [ cos -sin ]
    # [ sin  cos ]
    rot_matrix = np.array([
        [ math.cos(rad), -math.sin(rad)],
        [ math.sin(rad),  math.cos(rad)]
    ])
    
    # 4. Rotate the corner vectors
    rotated_corners = original_corners @ rot_matrix.T
    rotated_secondary_corners = secondary_corners @ rot_matrix.T
    rotated_tertiary_corners = tertiary_corners @ rot_matrix.T
    
    # 5. Shift to the new coordinate system
    # The new center is (rw/2, rh/2)
    new_cx, new_cy = rw / 2.0, rh / 2.0
    final_corners_1 = rotated_corners + np.array([new_cx, new_cy])
    final_corners_2 = rotated_secondary_corners + np.array([new_cx, new_cy])
    final_corners_3 = rotated_tertiary_corners + np.array([new_cx, new_cy])

    # resize the image to fit canvas if it's larger than canvas after rotation
    if rw > IMG_SIZE or rh > IMG_SIZE:
        scale_factor = min(IMG_SIZE * 0.9 / rw, IMG_SIZE * 0.9 / rh)
        rotated = rotated.resize((int(rw * scale_factor), int(rh * scale_factor)), Image.LANCZOS)
        final_corners_1 *= scale_factor
        final_corners_2 *= scale_factor
        final_corners_3 *= scale_factor
    
    return rotated, final_corners_1.astype(np.float32), final_corners_2.astype(np.float32), final_corners_3.astype(np.float32)




def move_image_to_edge(img: Image.Image, smallest_ydiff: float, smallest_xdiff: float):
    """
    Move the image to the edge of the canvas using only width and height.
    Pushes the image partially off-screen based on a random selection.
    """
    w, h = img.size
    # Pick a side/corner to push off-canvas
    # 0: Left, 1: Right, 2: Top, 3: Bottom
    side = random.randint(0, 3)
    
    # Define how much of the image we want to hide (10% to 40%)
    # Ensure the range for randint is valid (low <= high)
    off_x = smallest_xdiff * random.uniform(0.4, 0.8)
    off_y = smallest_ydiff * random.uniform(0.4, 0.8)

    # Calculate paste_x and paste_y, ensuring they are within valid ranges
    paste_x = 0
    paste_y = 0

    if side == 0:  # Clip Left side
        paste_x = -off_x
        paste_y = random.randint(0, max(0, IMG_SIZE - h))
        
    elif side == 1:  # Clip Right side
        paste_x = IMG_SIZE - w + off_x
        paste_y = random.randint(0, max(0, IMG_SIZE - h))
        
    elif side == 2:  # Clip Top side
        paste_x = random.randint(0, max(0, IMG_SIZE - w))
        paste_y = -off_y
        
    else:  # Clip Bottom side
        paste_x = random.randint(0, max(0, IMG_SIZE - w))
        paste_y = IMG_SIZE - h + off_y

    return int(paste_x), int(paste_y), side  # Return the side for visibility handling




def disk_convolution(img: Image.Image, radius: int) -> Image.Image:
    """
    Applies a disk convolution (blur) to the image.
    """
    # Create a disk kernel
    diameter = 2 * radius + 1
    Y, X = np.ogrid[-radius:radius+1, -radius:radius+1]
    mask = X**2 + Y**2 <= radius**2
    kernel = np.zeros((diameter, diameter), dtype=np.float32)
    kernel[mask] = 1
    kernel /= np.sum(kernel)

    # Convert image to numpy array for convolution
    img_np = np.array(img)

    # Apply convolution for each channel
    convolved_np = np.zeros_like(img_np, dtype=np.float32)
    for i in range(img_np.shape[2]): # Iterate over channels (RGBA)
        convolved_np[:, :, i] = ndimage.convolve(img_np[:, :, i], kernel, mode="nearest")
    
    # Convert back to PIL Image
    return Image.fromarray(convolved_np.astype(np.uint8))


def threshold_grayscale_to_black(img: Image.Image, threshold: int) -> Image.Image:
    """
    Converts pixels below a grayscale threshold to pure black.
    """
    # Convert to grayscale first if not already
    grayscale_img = img.convert("L")
    img_np = np.array(grayscale_img)

    # Apply threshold
    img_np[img_np < threshold] = 0
    
    # Convert back to PIL Image
    return Image.fromarray(img_np, mode="L").convert(img.mode)


# ---------- main generation routine ----------

def generate_sample(source_path: str, index: int):
    """
    Generate a single sample: transform source, create canvas, write labels
    and preview.  Returns (image_path, label_path) or (None, None) on failure.
    """
    # ---- 1.  load and transform  ----
    img = load_source(source_path)
    img = scale_to_canvas(img)

    angle = random.uniform(0, 85)
    img, mirrored, flip = apply_random_flips(img)
    img, local_corners_1, local_corner_2, local_corners_3 = rotate_image(img, angle, mirror=mirrored, flip=flip)   # local_corners_1 are rectangle corners of this image

    w_img, h_img = img.size

    # ---- 3.  paste onto black canvas (moved toward edge if corner cut) ----
    cut_idx = -1
    if 23 < angle < 65:
        # cut_idx = -1 means no cut, otherwise 0=L, 1=R, 2=T, 3=B
        # find top corner and bottom corner in local_corners_1 and their y diff with the closest point in y direction
        # find the left most corner and right most corner in local_corners_1 and their x diff with the closest point in x direction
        # find the smallest y diff
        # find the smallest x diff
        # sort local_corners_1 by y to find top and bottom corners
        marginy = [np.min(local_corners_1[:, 1]), h_img - np.max(local_corners_1[:, 1])]
        sorted_by_y = local_corners_1[np.argsort(local_corners_1[:, 1])]
        ydiff_arr = [abs(sorted_by_y[0, 1] - sorted_by_y[1,1]), abs(sorted_by_y[2, 1] - sorted_by_y[3, 1])] # y diff between top and bottom corners
        # include the margin space
        marginx = [np.min(local_corners_1[:, 0]), w_img - np.max(local_corners_1[:, 0])]
        sorted_by_x = local_corners_1[np.argsort(local_corners_1[:, 0])]
        xdiff_arr = [abs(sorted_by_x[0, 0] - sorted_by_x[1,0]), abs(sorted_by_x[2, 0] - sorted_by_x[3, 0])] # x diff between left and right corners
        
        xdiff_arr = [xdiff_arr[0] + marginx[0], xdiff_arr[1] + marginx[1]]
        ydiff_arr = [ydiff_arr[0] + marginy[0], ydiff_arr[1] + marginy[1]]
        smallest_ydiff = np.min(ydiff_arr)
        smallest_xdiff = np.min(xdiff_arr)
        paste_x, paste_y, cut_idx = move_image_to_edge(img, smallest_ydiff, smallest_xdiff)
    else:
        max_x = max(0, IMG_SIZE - w_img)
        max_y = max(0, IMG_SIZE - h_img)
        paste_x = random.randint(0, max_x) if max_x > 0 else 0
        paste_y = random.randint(0, max_y) if max_y > 0 else 0

    canvas = Image.new("RGBA", (IMG_SIZE, IMG_SIZE), (0, 0, 0, 255))
    canvas.paste(img, (paste_x, paste_y), img)

    # ---- 4.  compute visible bbox  ----
    # Bbox should now be the full pasted image, as we are not clipping.
    # So, bbox is effectively the image's dimensions at its paste location.
    bbox_min_x = min(local_corners_1[:, 0]) + paste_x
    bbox_min_y = min(local_corners_1[:, 1]) + paste_y
    bbox_max_x = max(local_corners_1[:, 0]) + paste_x
    bbox_max_y = max(local_corners_1[:, 1]) + paste_y
    # Clip bbox to canvas boundaries
    bbox_min_x = max(0, min(bbox_min_x, IMG_SIZE - 1))
    bbox_min_y = max(0, min(bbox_min_y, IMG_SIZE - 1))
    bbox_max_x = max(0, min(bbox_max_x, IMG_SIZE - 1))
    bbox_max_y = max(0, min(bbox_max_y, IMG_SIZE - 1))

    bbox_min_x_2 = min(local_corner_2[:, 0]) + paste_x
    bbox_min_y_2 = min(local_corner_2[:, 1]) + paste_y
    bbox_max_x_2 = max(local_corner_2[:, 0]) + paste_x
    bbox_max_y_2 = max(local_corner_2[:, 1]) + paste_y
    bbox_min_x_2 = max(0, min(bbox_min_x_2, IMG_SIZE - 1))
    bbox_min_y_2 = max(0, min(bbox_min_y_2, IMG_SIZE - 1))
    bbox_max_x_2 = max(0, min(bbox_max_x_2, IMG_SIZE - 1))
    bbox_max_y_2 = max(0, min(bbox_max_y_2, IMG_SIZE - 1))

    bbox_min_x_3 = min(local_corners_3[:, 0]) + paste_x
    bbox_min_y_3 = min(local_corners_3[:, 1]) + paste_y
    bbox_max_x_3 = max(local_corners_3[:, 0]) + paste_x
    bbox_max_y_3 = max(local_corners_3[:, 1]) + paste_y
    bbox_min_x_3 = max(0, min(bbox_min_x_3, IMG_SIZE - 1))
    bbox_min_y_3 = max(0, min(bbox_min_y_3, IMG_SIZE - 1))
    bbox_max_x_3 = max(0, min(bbox_max_x_3, IMG_SIZE - 1))
    bbox_max_y_3 = max(0, min(bbox_max_y_3, IMG_SIZE - 1))

    # If the image is completely outside or invisible after moving (shouldn't happen with correct logic)
    if bbox_max_x <= bbox_min_x or bbox_max_y <= bbox_min_y:
        return None, None
    
    if bbox_max_x_2 <= bbox_min_x_2 or bbox_max_y_2 <= bbox_min_y_2:
        return None, None
    
    if bbox_max_x_3 <= bbox_min_x_3 or bbox_max_y_3 <= bbox_min_y_3:
        return None, None

    # YOLO normalised bbox
    cx = ((bbox_min_x + bbox_max_x) / 2.0) / IMG_SIZE
    cy = ((bbox_min_y + bbox_max_y) / 2.0) / IMG_SIZE
    bw = (bbox_max_x - bbox_min_x) / IMG_SIZE
    bh = (bbox_max_y - bbox_min_y) / IMG_SIZE

    # YOLO normalised bbox
    cx_2 = ((bbox_min_x_2 + bbox_max_x_2) / 2.0) / IMG_SIZE
    cy_2 = ((bbox_min_y_2 + bbox_max_y_2) / 2.0) / IMG_SIZE
    bw_2 = (bbox_max_x_2 - bbox_min_x_2) / IMG_SIZE
    bh_2 = (bbox_max_y_2 - bbox_min_y_2) / IMG_SIZE

    # YOLO normalised bbox
    cx_3 = ((bbox_min_x_3 + bbox_max_x_3) / 2.0) / IMG_SIZE
    cy_3 = ((bbox_min_y_3 + bbox_max_y_3) / 2.0) / IMG_SIZE
    bw_3 = (bbox_max_x_3 - bbox_min_x_3) / IMG_SIZE
    bh_3 = (bbox_max_y_3 - bbox_min_y_3) / IMG_SIZE

    # ---- 5.  keypoints (4 corners of pasted image rectangle) ----
    # local_corners_1 is (N, 2) in local image space, top-left based
    # Convert to canvas space
    kp_canvas = local_corners_1 + np.array([paste_x, paste_y], dtype=np.float32)
    kp_canvas_2 = local_corner_2 + np.array([paste_x, paste_y], dtype=np.float32)
    kp_canvas_3 = local_corners_3 + np.array([paste_x, paste_y], dtype=np.float32)

    # Visibility status for each corner - all should be visible (2) since we move image instead of cutting
    visibilities = [2, 2, 2, 2]
    visibilities_2 = [2, 2, 2, 2]
    visibilities_3 = [2, 2, 2, 2]

    if cut_idx != -1:
        if cut_idx == 0:  # Left side cut
            #find left most corner in kp_canvas
            left_most_idx = np.argmin(kp_canvas[:, 0])
            visibilities[left_most_idx] = 1  # occluded
            # test if the corresponding corner in kp_canvas_2 and kp_canvas_3 is also occluded, if so, mark them as occluded too
            if kp_canvas_2[left_most_idx, 0] < 0:
                visibilities_2[left_most_idx] = 1  # occluded
            if kp_canvas_3[left_most_idx, 0] < 0:
                visibilities_3[left_most_idx] = 1  # occluded
        elif cut_idx == 1:  # Right side cut
            right_most_idx = np.argmax(kp_canvas[:, 0])
            visibilities[right_most_idx] = 1  # occluded
            # test if the corresponding corner in kp_canvas_2 and kp_canvas_3 is also occluded, if so, mark them as occluded too
            if kp_canvas_2[right_most_idx, 0] > IMG_SIZE:
                visibilities_2[right_most_idx] = 1  # occluded
            if kp_canvas_3[right_most_idx, 0] > IMG_SIZE:
                visibilities_3[right_most_idx] = 1  # occluded
        elif cut_idx == 2:  # Top side cut
            top_most_idx = np.argmin(kp_canvas[:, 1])
            visibilities[top_most_idx] = 1  # occluded
            if kp_canvas_2[top_most_idx, 1] < 0:
                visibilities_2[top_most_idx] = 1  # occluded
            if kp_canvas_3[top_most_idx, 1] < 0:
                visibilities_3[top_most_idx] = 1  # occluded
        elif cut_idx == 3:  # Bottom side cut
            bottom_most_idx = np.argmax(kp_canvas[:, 1])
            visibilities[bottom_most_idx] = 1  # occluded
            if kp_canvas_2[bottom_most_idx, 1] > IMG_SIZE:
                visibilities_2[bottom_most_idx] = 1  # occluded
            if kp_canvas_3[bottom_most_idx, 1] > IMG_SIZE:
                visibilities_3[bottom_most_idx] = 1  # occluded

    # adjust occuluded keypoints to be on the edge of the canvas
    for ci in range(4):
        if visibilities[ci] == 1:  # occluded
            kpx, kpy = kp_canvas[ci]
            kpx = max(0, min(kpx, IMG_SIZE - 1))
            kpy = max(0, min(kpy, IMG_SIZE - 1))
            kp_canvas[ci] = [kpx, kpy]

    for ci in range(4):
        if visibilities_2[ci] == 1:  # occluded
            kpx, kpy = kp_canvas_2[ci]
            kpx = max(0, min(kpx, IMG_SIZE - 1))
            kpy = max(0, min(kpy, IMG_SIZE - 1))
            kp_canvas_2[ci] = [kpx, kpy]

    for ci in range(4):
        if visibilities_3[ci] == 1:  # occluded
            kpx, kpy = kp_canvas_3[ci]
            kpx = max(0, min(kpx, IMG_SIZE - 1))
            kpy = max(0, min(kpy, IMG_SIZE - 1))
            kp_canvas_3[ci] = [kpx, kpy]

    # ---- 6.  write label file ----
    label_path = os.path.join(LABEL_DIR, f"usaf_full_{index:04d}.txt")
    with open(label_path, "w") as f:
        # Original order of keypoints is TL, TR, BR, BL from rotate_image.
        # The request is to have them start from top-left and go clockwise.
        # This is already the case: 0=TL, 1=TR, 2=BR, 3=BL.
        parts = [f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"]
        for ci in range(4):
            kx = kp_canvas[ci, 0] / IMG_SIZE
            ky = kp_canvas[ci, 1] / IMG_SIZE
            parts.append(f"{kx:.6f} {ky:.6f} {visibilities[ci]}")
        f.write(" ".join(parts) + "\n")

        parts = [f"0 {cx_2:.6f} {cy_2:.6f} {bw_2:.6f} {bh_2:.6f}"]
        for ci in range(4):
            kx = kp_canvas_2[ci, 0] / IMG_SIZE
            ky = kp_canvas_2[ci, 1] / IMG_SIZE
            parts.append(f"{kx:.6f} {ky:.6f} {visibilities_2[ci]}")
        f.write(" ".join(parts) + "\n")

        parts = [f"0 {cx_3:.6f} {cy_3:.6f} {bw_3:.6f} {bh_3:.6f}"]
        for ci in range(4):
            kx = kp_canvas_3[ci, 0] / IMG_SIZE
            ky = kp_canvas_3[ci, 1] / IMG_SIZE
            parts.append(f"{kx:.6f} {ky:.6f} {visibilities_3[ci]}")
        f.write(" ".join(parts) + "\n")

    # ---- 7.  save final RGB image ----
    # perform gaussian blur and thresholding to make it more realistic
    canvas = threshold_grayscale_to_black(canvas, threshold=30)
    canvas = ndimage.gaussian_filter(canvas, sigma=1.0)             # 1,3,5,7,9
    canvas = Image.fromarray(canvas.astype(np.uint8))
    canvas_rgb = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (0, 0, 0))
    mask = canvas.convert("L") # Use luminosity as mask
    canvas_rgb.paste(canvas, (0, 0), mask)

    image_path = os.path.join(IMAGE_DIR, f"usaf_full_{index:04d}.jpg")
    canvas_rgb.save(image_path, quality=95)

    # ---- 8.  draw preview  ----
    preview = canvas_rgb.copy()
    draw = ImageDraw.Draw(preview)

    # bounding box
    draw.rectangle([bbox_min_x, bbox_min_y, bbox_max_x, bbox_max_y],
                   outline=(255, 0, 0), width=1)
    draw.rectangle([bbox_min_x_2, bbox_min_y_2, bbox_max_x_2, bbox_max_y_2],
                   outline=(0, 255, 255), width=1)
    draw.rectangle([bbox_min_x_3, bbox_min_y_3, bbox_max_x_3, bbox_max_y_3],
                   outline=(255, 255, 0), width=1)
    
    # keypoints
    colors = [(0, 255, 0), (0, 255, 255), (255, 255, 0), (255, 0, 255)]
    colors_2 = [(255, 0, 255), (255, 255, 0), (0, 255, 255), (0, 255, 0)]
    colors_3 = [(0, 255, 255), (255, 255, 0), (255, 0, 255), (0, 255, 0)]
    for ci in range(4):
        kpx = int(kp_canvas[ci, 0])
        kpy = int(kp_canvas[ci, 1])
        if visibilities[ci] == 2:
            # visible: filled circle + label
            draw.ellipse([kpx - 1, kpy - 1, kpx + 1, kpy + 1],
                         fill=colors[ci], outline=(255, 100, 255), width=1)
            draw.text((kpx + 14, kpy - 8), str(ci + 1),
                      fill=(255, 255, 255))
        else:
            # occluded: red cross
            draw.line([kpx - 8, kpy - 8, kpx + 8, kpy + 8],
                      fill=(255, 0, 0), width=4)
            draw.line([kpx + 8, kpy - 8, kpx - 8, kpy + 8],
                      fill=(255, 0, 0), width=4)
            # label in red
            draw.text((kpx + 14, kpy - 8), str(ci + 1),
                      fill=(255, 100, 100))
            
    for ci in range(4):
        kpx = int(kp_canvas_2[ci, 0])
        kpy = int(kp_canvas_2[ci, 1])
        if visibilities_2[ci] == 2:
            # visible: filled circle + label
            draw.ellipse([kpx - 1, kpy - 1, kpx + 1, kpy + 1],
                         fill=colors_2[ci], outline=(255, 100, 255), width=1)
            draw.text((kpx + 14, kpy - 8), str(ci + 1),
                      fill=(255, 255, 255))
        else:
            # occluded: red cross
            draw.line([kpx - 8, kpy - 8, kpx + 8, kpy + 8],
                      fill=(255, 0, 0), width=4)
            draw.line([kpx + 8, kpy - 8, kpx - 8, kpy + 8],
                      fill=(255, 0, 0), width=4)
            # label in red
            draw.text((kpx + 14, kpy - 8), str(ci + 1),
                      fill=(255, 100, 100))
            
    for ci in range(4):
        kpx = int(kp_canvas_3[ci, 0])
        kpy = int(kp_canvas_3[ci, 1])
        if visibilities_3[ci] == 2:
            # visible: filled circle + label
            draw.ellipse([kpx - 1, kpy - 1, kpx + 1, kpy + 1],
                         fill=colors_3[ci], outline=(255, 100, 255), width=1)
            draw.text((kpx + 14, kpy - 8), str(ci + 1),
                      fill=(255, 255, 255))
        else:
            # occluded: red cross
            draw.line([kpx - 8, kpy - 8, kpx + 8, kpy + 8],
                      fill=(255, 0, 0), width=4)
            draw.line([kpx + 8, kpy - 8, kpx - 8, kpy + 8],
                      fill=(255, 0, 0), width=4)
            # label in red
            draw.text((kpx + 14, kpy - 8), str(ci + 1),
                      fill=(255, 100, 100))

    # corner index legend
    legend = "kp1(TL)  kp2(TR)  kp3(BR)  kp4(BL)   |   red cross=occluded"
    draw.text((10, 10), legend, fill=(200, 200, 200))

    preview_path = os.path.join(PREVIEW_DIR, f"usaf_full_{index:04d}_preview.jpg")
    preview.save(preview_path, quality=90)

    return image_path, label_path


def generate_dataset(source_path: str, num_samples: int = 100):
    """Generate *num_samples* synthetic images."""
    print(f"Starting generation of {num_samples} samples from {source_path}")
    if not os.path.isfile(source_path):
        raise FileNotFoundError(f"Source image not found: {source_path}")

    for i in range(num_samples):
        img_path, lbl_path = generate_sample(source_path, i)
        status = "OK" if img_path else "SKIP"
        print(f"[{i+1:3d}/{num_samples}] {status}  {img_path or ''}".strip())


# ---------- CLI ----------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Generate synthetic dataset for YOLOv8 keypoint detection."
    )
    parser.add_argument("source", type=str,
                        help="Path to the source image to transform.")
    parser.add_argument("--count", type=int, default=100,
                        help="Number of samples to generate (default: 100).")
    args = parser.parse_args()

    generate_dataset(args.source, args.count)
