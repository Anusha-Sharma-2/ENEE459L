from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont
import warnings
from scipy.ndimage import binary_propagation, convolve, gaussian_filter

MODEL_PATH = "version-RFB-320.onnx"

warnings.filterwarnings("ignore", category=UserWarning)

sess_options = ort.SessionOptions()
sess_options.log_severity_level = 3

# 3. Use only available providers to prevent the provider fallback warning
desired_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
available_providers = [p for p in desired_providers if p in ort.get_available_providers()]

# face detection
def hard_nms(box_scores, iou_threshold, top_k=-1, candidate_size=200):

    # Extract Values: Separate the input matrix into two components:
    # confidence scores and coordinate tuples (x1, y1, x2, y2)
    scores = box_scores[:, 4]
    boxes = box_scores[:, :4]

    # Sort by Confidence: Order the candidate bounding boxes in ascending order
    # according to their detection probability scores.
    indexes = np.argsort(scores)

    # Limit Candidates: Retain only the highest-scoring candidate indices up to
    # candidate_size to avoid unnecessary computation over low-probability background regions.
    indexes = indexes[-candidate_size:]

    picked = []

    # Iterative Suppression: While candidate indices remain:
    while len(indexes) > 0:

        # Extract the box associated with the highest remaining score
        # (the final index in the sorted list) and append it to the picked collection.
        current = indexes[-1]
        picked.append(current)

        # Terminate the loop immediately if the total number of picked boxes
        # reaches top_k or if only one candidate remains
        if (top_k > 0 and len(picked) >= top_k) or len(indexes) == 1:
            break

        # Compare this designated box against all remaining candidate boxes by
        # calculating their Intersection over Union (IoU):

        # Determine the coordinates of the intersecting rectangle between the active box and each remaining box.
        current_box = boxes[current]

        remaining = indexes[:-1]
        remaining_boxes = boxes[remaining]

        xmin = np.maximum(current_box[0], remaining_boxes[:, 0])
        ymin = np.maximum(current_box[1], remaining_boxes[:, 1])
        xmax = np.minimum(current_box[2], remaining_boxes[:, 2])
        ymax = np.minimum(current_box[3], remaining_boxes[:, 3])


        intersection = (
            np.maximum(0, xmax - xmin) *
            np.maximum(0, ymax - ymin)
        )

        area_current = (
            (current_box[2] - current_box[0]) *
            (current_box[3] - current_box[1])
        )

        area_remaining = (
            (remaining_boxes[:, 2] - remaining_boxes[:, 0]) *
            (remaining_boxes[:, 3] - remaining_boxes[:, 1])
        )

        # Compute the combined union area:
        union = area_current + area_remaining - intersection

        # Determine the ratio:
        iou = intersection / np.maximum(union, 1e-10)

        # Discard all candidate boxes whose IoU exceeds iou_threshold.
        indexes = remaining[iou <= iou_threshold]

    # Return Filtered Subset: Return the rows of the original array matching the indices retained in picked.
    return box_scores[picked, :]

def detect_faces(image_path, output_path="output_detected.png", conf_threshold=0.7):
    # Construct an end-to-end inference routine that ingests an image file from storage, normalizes it
    # into tensor format, executes forward inference via ONNX Runtime using an ultra-lightweight
    # detection model, applies Non-Maximum Suppression to consolidate outputs, and renders labeled
    # bounding boxes on the original image

    # Load and Record Geometry: Open the target image using Pillow, ensure 3-channel
    image = Image.open(image_path).convert("RGB")
    orig_width, orig_height = image.size

    # RGB formatting, and record the original width and height for post-inference scaling.

    # Preprocess Input Tensor:
    # • Resize the image array to match the input specification of the network (320 × 240).
    # • Convert integer pixel intensities to 32-bit floating-point format.
    # • Normalize pixel values to the range [−1.0, 1.0] via the transformation:
    image_resized = image.resize((320, 240))
    x = np.asarray(image_resized, dtype=np.float32)
    x_norm = (x - 127.0) / 128.0

    # transpose data from Height-Width-Channel (H, W, C) to Channel-Height-Width (C, H, W), then prepend an explicit batch dimension: (1, C, H, W).
    x_norm = np.transpose(x_norm, (2, 0, 1))
    x_norm = np.expand_dims(x_norm, axis=0)

    # Execute Inference Session: Instantiate an ONNX Runtime session specifying available
    # execution providers (such as CUDA or CPU) and evaluate the graph using the preprocessed array.
    model_path = Path(__file__).with_name(MODEL_PATH)
    session = ort.InferenceSession(
        str(model_path),
        sess_options=sess_options,
        providers=available_providers
    )
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: x_norm})

    # Filter and Suppress Candidates:
    # • Remove the batch dimension from output tensors.
    # • Mask out detections whose confidence falls below conf_threshold.
    # • Pass surviving coordinates and scores to hard_nms to eliminate overlapping duplicate boxes.
    scores = outputs[0][0]
    boxes = outputs[1][0]

    if scores.ndim == 2:
        scores = scores[:, 1]

    mask = scores >= conf_threshold
    boxes = boxes[mask]
    scores = scores[mask]

    box_scores = np.concatenate(
        (boxes, scores.reshape(-1, 1)),
        axis=1
    )

    picked = hard_nms(
        box_scores,
        iou_threshold=0.45,
        top_k=-1
    )

    # Annotation and Export:
    # • Scale normalized bounding box coordinates back to full image dimensions by multiplying by original width and height.
    # • Draw rectangular outlines and confidence text overlays on the image canvas.
    # • Save the annotated graphic to the destination filepath.

    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except OSError:
        font = ImageFont.load_default()

    for detection in picked:
        x1, y1, x2, y2, confidence = detection
        x1 *= orig_width
        x2 *= orig_width
        y1 *= orig_height
        y2 *= orig_height
        draw.rectangle(
            [x1, y1, x2, y2],
            outline="lime",
            width=3
        )

        label = f"Face: {confidence:.2f}"

        draw.text(
            (x1, max(0, y1 - 20)),
            label,
            fill="lime",
            font=font
        )

    image.save(output_path)

    return image

# Edge detection
def non_maximum_suppression(magnitude, angle):
    magnitude = np.asarray(magnitude)
    angle = np.asarray(angle)
    if magnitude.ndim != 2 or angle.shape != magnitude.shape:
        raise ValueError("magnitude and angle must be same-sized 2D arrays")

    angle_degrees = np.rad2deg(angle) % 180.0
    thinned = np.zeros(magnitude.shape, dtype=np.float32)

    height, width = magnitude.shape
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            direction = angle_degrees[i, j]
            if direction < 22.5 or direction >= 157.5:
                neighbors = magnitude[i, j - 1], magnitude[i, j + 1]
            elif direction < 67.5:
                neighbors = magnitude[i + 1, j - 1], magnitude[i - 1, j + 1]
            elif direction < 112.5:
                neighbors = magnitude[i - 1, j], magnitude[i + 1, j]
            else:
                neighbors = magnitude[i - 1, j - 1], magnitude[i + 1, j + 1]

            if magnitude[i, j] >= neighbors[0] and magnitude[i, j] >= neighbors[1]:
                thinned[i, j] = magnitude[i, j]

    return thinned

def hysteresis_thresholding(img, low_thresh, high_thresh):
    img = np.asarray(img)
    if img.ndim != 2:
        raise ValueError("img must be a 2D array")
    if low_thresh > high_thresh:
        raise ValueError("low_thresh must not exceed high_thresh")

    strong = img >= high_thresh
    weak = (img >= low_thresh) & ~strong
    connected = binary_propagation(
        strong,
        structure=np.ones((3, 3), dtype=bool),
        mask=strong | weak
    )
    return (connected * 255).astype(np.uint8)

def canny_scipy(image_path, output_path="edges_scipy.png", low=20, high=60):
    image = Image.open(image_path).convert("L")
    grayscale = np.asarray(image, dtype=np.float32)
    smoothed = gaussian_filter(grayscale, sigma=1.4)

    sobel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float32)
    sobel_y = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]
    ], dtype=np.float32)

    gradient_x = convolve(smoothed, sobel_x)
    gradient_y = convolve(smoothed, sobel_y)
    magnitude = np.hypot(gradient_x, gradient_y)
    max_magnitude = magnitude.max()
    if max_magnitude > 0:
        magnitude = magnitude * (255.0 / max_magnitude)

    angle = np.arctan2(gradient_y, gradient_x)
    thinned = non_maximum_suppression(magnitude, angle)
    edges = hysteresis_thresholding(thinned, low, high)

    result = Image.fromarray(edges)
    result.save(output_path)
    return result

if __name__ == "__main__":
    detect_faces("face.jpg")
    canny_scipy("face.jpg")