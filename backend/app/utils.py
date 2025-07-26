from pdf2image import convert_from_path
from PIL import Image, ImageOps
import numpy as np
import cv2
import os

def get_line_coords(lines, color_overlay, debug):
    # Extract bounding boxes of underscore lines
    contours, _ = cv2.findContours(lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    line_coords = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w >= 30: 
            line_coords.append((x, y, w, h))
            label = f"({x},{y})"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = .5
            thickness = 1

            # Put text just above the line, aligned to left corner
            text_x = x
            text_y = y - 5 if y - 5 > 0 else y + 10  # Avoid going above image
            cv2.putText(color_overlay, label, (text_x, text_y), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
    
    return line_coords


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
    binary_no_lines = cv2.bitwise_not(binary) #TODO: Decided to remove all lines in preprocessing not just summing lines since I didnt want OCR to capture them as somethjing else. IF this causes errors, then change it back
    no_lines = cv2.bitwise_or(binary_no_lines, all_line_segments)
    cleaned = cv2.bitwise_not(no_lines)

    # Remove small noise blobs
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8) 
    filtered = np.zeros_like(cleaned)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= 20:
            filtered[labels == i] = 255

    final_image = Image.fromarray(cv2.cvtColor(filtered, cv2.COLOR_GRAY2RGB))

    underscore_zero_coords = get_line_coords(filtered_underscores, color_overlay, debug)
    summing_line_coords = get_line_coords(summing_lines, color_overlay, debug)

    debug_path = os.path.join("Debug", "debug_detected_lines.png")
    Image.fromarray(color_overlay).save(debug_path)
    print(f"Debug image saved to: {debug_path}")

    return final_image, underscore_zero_coords, summing_line_coords


def get_coords(bbox):
        """
        Unpacks bbox coordinates from OCR results
        """
        tl, tr, bl, _ = bbox
        x1, y1 = tl
        x2, _ = tr
        _, y2 = bl
        return (x1, x2, y1, y2)


def get_all_subsets(lst):
    """
    Recursively finds all subsets of a given list
    """
    subsets = []

    def backtrack(start, path):
        if path:
            subsets.append(path[:])  # Add a copy of the current subset
        for i in range(start, len(lst)):
            backtrack(i + 1, path + [lst[i]])

    backtrack(0, [])
    return subsets


def find_solution(subsets, target=0, off_by_thresh=0, penalize_indices=None):
    """
    Finds the best subset and +/- sign combo that sums to target.
    Prefers exact matches, and scores them. Falls back to near matches if needed.
    """
    if penalize_indices is None:
        penalize_indices = set()

    exact_matches = []
    near_matches = []

    def try_signs(subset):
        n = len(subset)
        results = []

        def recurse(i, current_sum, current_combo):
            if i == n:
                off_by = abs(current_sum - target)
                if off_by <= off_by_thresh:
                    results.append((current_combo[:], off_by))
                return

            index, value = subset[i]

            # Try adding
            recurse(i + 1, current_sum + value, current_combo + [(index, True)])
            # Try subtracting
            recurse(i + 1, current_sum - value, current_combo + [(index, False)])

        recurse(0, 0, [])
        return results

    for subset in subsets:
        for combo, off_by in try_signs(subset):
            if off_by == 0:
                exact_matches.append(combo)
            else:
                near_matches.append((combo, off_by))

    def score(combo):
        indices = [i for i, _ in combo]
        num_adds = sum(1 for _, sign in combo if sign)
        num_subs = sum(1 for _, sign in combo if not sign)
        subtotal_penalty = sum(1 for i in indices if i in penalize_indices)

        return (
            len(combo) * 10             # prefer using more values
            + num_adds * 5              # prefer addition
            - num_subs * 2              # slightly penalize subtraction
            - subtotal_penalty * 50     # heavily penalize use of subtotal lines
        )

    # Score and return best exact match if any
    if exact_matches:
        return max(exact_matches, key=score)

    # Otherwise, score and return best near match
    if near_matches:
        return max((combo for combo, _ in near_matches), key=score)

    return None  # No valid solution found
