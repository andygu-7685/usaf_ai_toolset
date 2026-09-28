from pathlib import Path
import cv2
from ultralytics import YOLO

import matplotlib.pyplot as plt
import numpy as np


def rotate_image(image, angle):
    """
    Rotates an image without clipping the corners by expanding the canvas.
    """
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # 1. Get the rotation matrix
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)

    # 2. Calculate the sine and cosine of the angle
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # 3. Compute the new bounding dimensions
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # 4. Adjust the rotation matrix to take the translation into account
    # This ensures the image stays centered in the new larger canvas
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # 5. Perform the rotation with the new dimensions
    return cv2.warpAffine(image, M, (nW, nH), borderValue=(0, 0, 0))


def preprocess_to_bw(image, threshold: int = 140, invert: bool = False, to_bgr: bool = True):
    """
    Convert an image to pure black/white using a grayscale threshold.

    Args:
        image: OpenCV image (BGR) as a numpy array.
        threshold: 0-255 threshold value.
        invert: If True, pixels below threshold become white instead.
        to_bgr: If True, returns a 3-channel BGR image (so it's model/display compatible).

    Returns:
        BGR image if to_bgr else single-channel grayscale.
    """
    if image is None:
        raise ValueError("image is None")

    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    thresh_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, bw = cv2.threshold(gray, threshold, 255, thresh_type)

    if to_bgr:
        return cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
    return bw


def main():
    model_path = Path("models\\best_zoom_3.pt")
    if not model_path.exists():
        raise FileNotFoundError("Could not find trained YOLOv8 model at runs/detect/train/weights/best.pt or best.pt")

    test_dir = Path("test")
    if not test_dir.exists():
        raise FileNotFoundError(f"Test folder not found: {test_dir}")

    image_files = sorted(test_dir.glob("*.*"))
    if not image_files:
        raise FileNotFoundError(f"No images found in {test_dir}")

    image_path = image_files[0]
    print(f"Using image: {image_path}")

    # pad the image with black margin for better detecion
    image = cv2.imread(str(image_path))
    # image = cv2.copyMakeBorder(image, 400, 400, 0, 400, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    # image = cv2.resize(image, (640,640), interpolation=cv2.INTER_AREA)

    kernel = np.array([ [-2, -2, -2],
                        [-2, 18, -2],
                        [-2, -2, -2]], dtype=np.float32)

    image = cv2.filter2D(image, -1, kernel)
    # perform clahe from cv2
    # clahe = cv2.createCLAHE(clipLimit=7.0, tileGridSize=(28, 28))
    # convert to grayscale
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # image = clahe.apply(image)
    # convert back to bgr
    #image = cv2.cvtColor(image, cv2.COLOR_ GRAY2BGR)
    # preprocess the image to black/white
    # image = preprocess_to_bw(image)

    image = cv2.resize(image, (160,160), interpolation=cv2.INTER_AREA)

    image = rotate_image(image, 15)

    bordered_img = cv2.copyMakeBorder(image, 500, 500, 500, 500, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    model = YOLO(str(model_path))
    results = model(image, imgsz=260, iou=0.2, conf=0.25)
    result = results[0]

    # Draw annotations using only OpenCV, with 1px line width for boxes and points.
    annotated = image.copy()

    thickness = 1
    
    box_color = (0, 255, 255)   # yellow/cyan in BGR
    kp_color = (0, 255, 0)       # green in BGR
    font = cv2.FONT_HERSHEY_SIMPLEX

    # result.boxes: xyxy, cls, conf
    if result.boxes is not None and len(result.boxes) > 0 and result.keypoints is not None:
        xyxy = result.boxes.xyxy.cpu().numpy().astype(int)   # (N, 4)
        cls = result.boxes.cls.cpu().numpy().astype(int)      # (N,)
        conf = result.boxes.conf.cpu().numpy().astype(float)  # (N,)

        kps_xy = result.keypoints.xy.cpu().numpy().astype(int)  # (N, K, 2)
        kps_conf = None
        if getattr(result.keypoints, "conf", None) is not None:
            kps_conf = result.keypoints.conf.cpu().numpy().astype(float)  # (N, K)

        for i in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[i]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, thickness=thickness)

            # Optional label (keep thickness=1 so it doesn't dominate).
            label = f"{cls[i]} {conf[i]:.2f}"
            # cv2.putText(annotated, label, (x1, max(0, y1 - 3)),
            #             font, 0.35, box_color, thickness)

            for kp_j in range(kps_xy.shape[1]):
                x, y = int(kps_xy[i, kp_j, 0]), int(kps_xy[i, kp_j, 1])

                # Many pose heads use (0,0) for "not present".
                if x == 0 and y == 0:
                    continue
                if kps_conf is not None and kps_conf[i, kp_j] <= 0.0:
                    continue

                # 1px "point": draw a small cross using 1px lines (no filled circles).
                s = 2
                cv2.line(annotated, (x - s, y), (x + s, y), kp_color, thickness=thickness)
                cv2.line(annotated, (x, y - s), (x, y + s), kp_color, thickness=thickness)

    annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(12, 8))
    plt.imshow(annotated)
    plt.axis("off")
    plt.title(f"YOLOv8 result: {image_path.name}")
    plt.show()

if __name__ == "__main__":
    main()