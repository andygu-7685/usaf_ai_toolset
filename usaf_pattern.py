import random
import math
from PIL import Image, ImageDraw
import os
import numpy as np
import torch
import torch.nn.functional as F
import cv2

# Create folders
os.makedirs("usaf_dataset/images", exist_ok=True)
os.makedirs("usaf_dataset/labels", exist_ok=True)
os.makedirs("preview", exist_ok=True)

# Image size
IMG_SIZE = 2048
CONVO = True


def gaussian_blur_kernel(size=5, sigma=1.0):
    """Generate a Gaussian blur kernel."""
    ax = np.arange(-size // 2 + 1., size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    return kernel / np.sum(kernel)


def disk_kernel(radius):
    kernel = np.zeros((2*radius+1, 2*radius+1), np.float32)
    # Draw a filled white circle on a black background
    cv2.circle(kernel, (radius, radius), radius, 1, -1)
    # Normalize so the sum is 1
    return kernel / kernel.sum()


def convolve2d(image, kernel, padding=0, stride=1):
    """
    Perform 2D convolution on an image with GPU acceleration using PyTorch.
    
    Args:
        image: Input image as numpy array (H x W)
        kernel: Convolution kernel (K x K)
        padding: Number of pixels to pad (default: 0)
        stride: Stride of convolution (default: 1)
    
    Returns:
        Convolved image as numpy array
    """
    # Check GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Convert numpy arrays to PyTorch tensors
    image_tensor = torch.from_numpy(image).float().to(device)
    kernel_tensor = torch.from_numpy(kernel).float().to(device)
    
    # Add batch and channel dimensions for conv2d: (1, 1, H, W)
    image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)
    kernel_tensor = kernel_tensor.unsqueeze(0).unsqueeze(0)
    
    # Apply convolution using PyTorch's conv2d
    output = F.conv2d(image_tensor, kernel_tensor, padding=padding, stride=stride)
    
    # Remove batch and channel dimensions and convert back to numpy
    output = output.squeeze(0).squeeze(0)
    output = output.cpu().numpy()
    
    # Normalize to [0, 255]
    output = np.clip(output, 0, 255)
    return output




def generate_usaf_pattern(index, kernel_type=1):
    img = Image.new('RGB', (IMG_SIZE, IMG_SIZE), (0, 0, 0))
    bboxes = []
    keypoints = []
    num_patterns = 50
    attempts = 0
    max_attempts = 1000

    while len(bboxes) < num_patterns and attempts < max_attempts:
        attempts += 1
        # Random size between 5 and 200
        size = random.randint(5, 100)
        # Random angle 0 to 360
        angle = random.uniform(0, 360)
        # Random center
        center_x = random.randint(0, IMG_SIZE)
        center_y = random.randint(0, IMG_SIZE)
        
        # Compute bounding box
        half = size / 2
        corners = [
            (-half, -half),
            (half, -half),
            (half, half),
            (-half, half)
        ]
        rad = math.radians(angle)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        rot_corners = []
        for x, y in corners:
            rx = x * cos_a - y * sin_a
            ry = x * sin_a + y * cos_a
            rot_corners.append((rx, ry))
        min_x = min(rx for rx, ry in rot_corners) + center_x
        max_x = max(rx for rx, ry in rot_corners) + center_x
        min_y = min(ry for rx, ry in rot_corners) + center_y
        max_y = max(ry for rx, ry in rot_corners) + center_y
        
        # Check bounds
        if min_x < 0 or max_x >= IMG_SIZE or min_y < 0 or max_y >= IMG_SIZE:
            continue
        
        # Check overlap
        overlap = False
        for bx_min, by_min, bx_max, by_max in bboxes:
            if not (max_x < bx_min or min_x > bx_max or max_y < by_min or min_y > by_max):
                overlap = True
                break
        if overlap:
            continue
        
        # Add to bboxes
        bboxes.append((min_x, min_y, max_x, max_y))
        
        # Draw the pattern
        pattern_img = Image.new('RGBA', (int(size * 2), int(size * 2)), (0, 0, 0, 0))
        draw = ImageDraw.Draw(pattern_img)
        bar_width = size / 5
        offset = size / 2
        colors = [(255, 255, 255, 255), (0, 0, 0, 255), (255, 255, 255, 255), (0, 0, 0, 255), (255, 255, 255, 255)]
        for i in range(5):
            x = i * bar_width + offset
            draw.rectangle([x, offset, x + bar_width, size + offset], fill=colors[i])
        
        # Rotate
        rotated = pattern_img.rotate(angle, expand=True)
        
        # Paste
        rot_width, rot_height = rotated.size
        paste_x = int(center_x - rot_width / 2)
        paste_y = int(center_y - rot_height / 2)
        img.paste(rotated, (paste_x, paste_y), rotated)

        # Compute scanline keypoints across the 3 bars (Horizontal Scan)
        # We move from the center of the first bar to the center of the last bar

        half = size / 2
        bar_width = size / 5

        # Point 1: Center of the LEFTMOST bar
        p1_x_local = -half + (bar_width / 2)
        p1_y_local = 0 

        # Point 2: Center of the RIGHTMOST bar
        p2_x_local = -half + (4.5 * bar_width)
        p2_y_local = 0

        # acccount for PIL rotation direction (negative angle)
        cos_a = math.cos(-rad)
        sin_a = math.sin(-rad)

        # Rotation Math (remains the same)
        rx1 = p1_x_local * cos_a - p1_y_local * sin_a
        ry1 = p1_x_local * sin_a + p1_y_local * cos_a

        rx2 = p2_x_local * cos_a - p2_y_local * sin_a
        ry2 = p2_x_local * sin_a + p2_y_local * cos_a

        # Global Translation (remains the same)
        abs_k1x = center_x + rx1
        abs_k1y = center_y + ry1
        abs_k2x = center_x + rx2
        abs_k2y = center_y + ry2

        keypoints.append((abs_k1x, abs_k1y, abs_k2x, abs_k2y))



















    # Add random rotated white squares as noise (non-overlapping with patterns)
    draw = ImageDraw.Draw(img)
    num_noise_boxes = random.randint(10, 30)
    noise_added = 0
    noise_attempts = 0
    max_noise_attempts = 500
    
    while noise_added < num_noise_boxes and noise_attempts < max_noise_attempts:
        noise_attempts += 1
        noise_size = random.randint(10, 80)
        noise_angle = random.uniform(0, 360)
        noise_x = random.randint(0, IMG_SIZE)
        noise_y = random.randint(0, IMG_SIZE)
        
        # Create rotated square
        noise_img = Image.new('RGBA', (noise_size, noise_size), (0, 0, 0, 0))
        draw_noise = ImageDraw.Draw(noise_img)
        draw_noise.rectangle([0, 0, noise_size, noise_size], fill=(255, 255, 255, 255))
        
        # Rotate the noise square
        rotated_noise = noise_img.rotate(noise_angle, expand=True)
        rot_width, rot_height = rotated_noise.size
        paste_x = int(noise_x - rot_width / 2)
        paste_y = int(noise_y - rot_height / 2)
        
        # Check bounds
        if paste_x < 0 or paste_x + rot_width >= IMG_SIZE or paste_y < 0 or paste_y + rot_height >= IMG_SIZE:
            continue
        
        # Check overlap with patterns
        noise_min_x = paste_x
        noise_max_x = paste_x + rot_width
        noise_min_y = paste_y
        noise_max_y = paste_y + rot_height
        
        overlap = False
        for bx_min, by_min, bx_max, by_max in bboxes:
            if not (noise_max_x < bx_min or noise_min_x > bx_max or noise_max_y < by_min or noise_min_y > by_max):
                overlap = True
                break
        
        if overlap:
            continue
        
        # Paste the rotated noise square
        img.paste(rotated_noise, (paste_x, paste_y), rotated_noise)
        noise_added += 1








    # Draw bounding boxes on a copy for review
    img_with_boxes = img.copy()
    draw_img = ImageDraw.Draw(img_with_boxes)
    for min_x, min_y, max_x, max_y in bboxes:
        draw_img.rectangle([min_x, min_y, max_x, max_y], outline=(255, 0, 0), width=2)
    for kp in keypoints:
        draw_img.line([kp[0], kp[1], kp[2], kp[3]], fill=(0, 255, 0), width=2)



    img_array = np.array(img, dtype=np.float32)
    # Convert to grayscale for convolution if RGB
    if len(img_array.shape) == 3:
        img_array = np.mean(img_array, axis=2)
    
    # 5x5 Blur kernel for smoothing
    blur_kernel1 = np.array([[1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1], 
                            [1, 1, 1, 1, 1], 
                            [1, 1, 1, 1, 1], 
                            [1, 1, 1, 1, 1]], dtype=np.float32) / 25.0
    
    # 5x5 gaussian blur kernel for smoothing
    blur_kernel2 = np.array([[1, 4, 7, 4, 1],
                            [4, 16, 26, 16, 4], 
                            [7, 26, 41, 26, 7], 
                            [4, 16, 26, 16, 4], 
                            [1, 4, 7, 4, 1]], dtype=np.float32) / 273.0
    
    # 5x5 anti-gaussian blur kernel for smoothing
    blur_kernel3 = (41/273.0 - blur_kernel2) * 273.0 / 752.0

    # 3x3 average blur kernel for smoothing
    blur_kernel4 = np.array([[1, 1, 1],
                            [1, 1, 1], 
                            [1, 1, 1]], dtype=np.float32) / 9.0
    
    # 3x3 gaussian blur kernel for smoothing
    blur_kernel5 = np.array([[1, 2, 1],
                            [2, 4, 2], 
                            [1, 2, 1]], dtype=np.float32) / 16.0
    
    # 3x3 anti-gaussian blur kernel for smoothing
    blur_kernel6 = (4/16.0 - blur_kernel5) * 16.0 / 9.0

    blur_kernel7 = np.array([[1, 1, 1, 1, 1],
                             [1, 2, 2, 2, 1], 
                             [1, 2, 20, 2, 1], 
                             [1, 2, 2, 2, 1], 
                             [1, 1, 1, 1, 1]], dtype=np.float32) / 52.0
    
    blur_kernel8 = gaussian_blur_kernel(size=5, sigma=7.0)

    blur_kernel9 = disk_kernel(radius=4)

    # identity kernel (no blur)
    identity_kernel = np.array([[0, 0, 0],
                                [0, 1, 0], 
                                [0, 0, 0]], dtype=np.float32)

    # Apply convolution with padding
    if kernel_type == 1:
        convolved = convolve2d(img_array, blur_kernel1, padding=2)
    elif kernel_type == 2:
        convolved = convolve2d(img_array, blur_kernel2, padding=1)
    elif kernel_type == 3:
        convolved = convolve2d(img_array, blur_kernel3, padding=1)
    elif kernel_type == 4:
        convolved = convolve2d(img_array, blur_kernel4, padding=1)
    elif kernel_type == 5:
        convolved = convolve2d(img_array, blur_kernel5, padding=1)
    elif kernel_type == 6:
        convolved = convolve2d(img_array, blur_kernel6, padding=1)
    elif kernel_type == 7:
        convolved = convolve2d(img_array, blur_kernel7, padding=2)
    elif kernel_type == 8:
        convolved = convolve2d(img_array, blur_kernel8, padding=5)
    elif kernel_type == 9:
        convolved = convolve2d(img_array, blur_kernel9, padding=4)
    else:
        convolved = convolve2d(img_array, identity_kernel, padding=1)
    # Save convolved image
    if CONVO:
        img = Image.fromarray(np.uint8(convolved))
    else:
        img = Image.fromarray(np.uint8(img_array))



    # Save images
    img.save(f"usaf_dataset/images/usaf_pattern_{index}.jpg")
    img_with_boxes.save(f"preview/usaf_pattern_{index}_with_boxes.jpg")

    # Save labels in YOLOv8-Pose format
    label_path = f"usaf_dataset/labels/usaf_pattern_{index}.txt"

    with open(label_path, "w") as f:
        for i in range(len(bboxes)):
            # 1. Get Bounding Box coordinates
            min_x, min_y, max_x, max_y = bboxes[i]
            
            # 2. Calculate YOLO center-relative coordinates (normalized 0-1)
            x_center = ((min_x + max_x) / 2) / IMG_SIZE
            y_center = ((min_y + max_y) / 2) / IMG_SIZE
            width    = (max_x - min_x) / IMG_SIZE
            height   = (max_y - min_y) / IMG_SIZE
            
            # 3. Get and Normalize Keypoints
            k1x, k1y, k2x, k2y = keypoints[i]

            # Order points by the dominant axis:
            # - if y difference is larger, k1 is the topmost point
            # - if x difference is larger, k1 is the leftmost point
            dy = abs(k2y - k1y)
            dx = abs(k2x - k1x)
            if dy > dx:
                if k2y < k1y:
                    k1x, k1y, k2x, k2y = k2x, k2y, k1x, k1y
            elif dx > dy:
                if k2x < k1x:
                    k1x, k1y, k2x, k2y = k2x, k2y, k1x, k1y
            else:
                # if both differences are equal, fall back to topmost then leftmost
                if k2y < k1y or (k2y == k1y and k2x < k1x):
                    k1x, k1y, k2x, k2y = k2x, k2y, k1x, k1y
            
            # Use a small epsilon or clip to ensure values stay within [0, 1] for Roboflow
            k1x_n = max(0, min(1, k1x / IMG_SIZE))
            k1y_n = max(0, min(1, k1y / IMG_SIZE))
            k2x_n = max(0, min(1, k2x / IMG_SIZE))
            k2y_n = max(0, min(1, k2y / IMG_SIZE))
            
            # 4. Visibility flags: 
            # 0 = not labeled, 1 = labeled but occluded, 2 = labeled and visible
            v1, v2 = 2, 2 
            
            # 5. Write to file
            # Format: <class> <x> <y> <w> <h> <px1> <py1> <pv1> <px2> <py2> <pv2>
            line = (f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} "
                    f"{k1x_n:.6f} {k1y_n:.6f} {v1} {k2x_n:.6f} {k2y_n:.6f} {v2}\n")
            f.write(line)


kernel_seq = [-1, 5, 9]

for i in range(100):
    generate_usaf_pattern(i, kernel_seq[i % len(kernel_seq)])