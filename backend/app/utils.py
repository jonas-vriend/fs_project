from pdf2image import convert_from_path
from PIL import Image, ImageOps
import numpy as np
import cv2
import os


def preprocess_img(pdf_path, debug):
    """
    Converts PDF to image and enhances it for OCR.
    Detects and removes summing lines.
    Identifies underscore lines (potential 0s) not overlapping summing lines.
    Returns: cleaned image, debug path, underscore_coords
    """
    # Convert PDF to grayscale image
    image = convert_from_path(pdf_path, dpi=300)
    gray = image[0].convert("L")
    contrast_img = ImageOps.autocontrast(gray)

    # Scale for better OCR
    scale_factor = 2
    resized = contrast_img.resize((gray.width * scale_factor, gray.height * scale_factor))
    img = np.array(resized)

    # Threshold and invert for line detection
    _, binary = cv2.threshold(img, 160, 255, cv2.THRESH_BINARY_INV)

    # Identify summing lines and underscore candidates

    line_segmnet_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (70, 1)) # creates the kernel
    summing_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, line_segmnet_kernel, iterations=2) #binary img showing lines (TODO)
    all_line_segments = cv2.morphologyEx(binary, cv2.MORPH_OPEN, line_segmnet_kernel, iterations=1) #binary img showing underscores + lines (TODO)

    # Isolate underscore
    filtered_underscores = cv2.subtract(all_line_segments, summing_lines)

    # Create debug overlay
    color_overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) 
    color_overlay[summing_lines > 0] = [255, 0, 0]  # Red for summing lines
    color_overlay[filtered_underscores > 0] = [0, 255, 0]  # Green for underscore 0s

    # Remove lines from image 
    binary_no_summing = cv2.bitwise_not(binary)
    no_summing = cv2.bitwise_or(binary_no_summing, summing_lines)
    cleaned = cv2.bitwise_not(no_summing)

    # Remove small noise blobs
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8) 
    filtered = np.zeros_like(cleaned)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= 20:
            filtered[labels == i] = 255

    final_image = Image.fromarray(cv2.cvtColor(filtered, cv2.COLOR_GRAY2RGB))

    # Extract bounding boxes of underscore lines
    contours, _ = cv2.findContours(filtered_underscores, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    underscore_coords = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w >= 30: 
            underscore_coords.append((x, y, w, h))
            if debug:
                print(f'Captured line: X: {x} Y: {y}')
            label = f"({x},{y})"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = .5
            thickness = 1

            # Put text just above the line, aligned to left corner
            text_x = x
            text_y = y - 5 if y - 5 > 0 else y + 10  # Avoid going above image
            cv2.putText(color_overlay, label, (text_x, text_y), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

    # Save debug overlay 
    debug_path = os.path.join("Debug", "debug_detected_lines.png")
    Image.fromarray(color_overlay).save(debug_path)
    print(f"Debug image saved to: {debug_path}")

    return final_image, underscore_coords

def get_coords(bbox):
        """
        Unpacks bbox coordinates from OCR results
        """
        tl, tr, bl, _ = bbox
        x1, y1 = tl
        x2, _ = tr
        _, y2 = bl
        return (x1, x2, y1, y2)


def get_x_bounds(list_of_coords, num_years):
    """
    Takes list of coordinates of detected values in each row and takes median value
    of the right bbox border x values for each column as a guide for where the values should be
    """
    cols = [[] for _ in range(num_years)]

    for coords in list_of_coords:
        if not len(coords) == num_years:
            continue
        for i, coord in enumerate(coords):
            cols[i].append(coord)

    col_coords = [np.median(col) for col in cols]
    return col_coords