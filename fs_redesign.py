import easyocr
import cv2
import re
import pickle
import os
import csv
import numpy as np
from collections import Counter
from pdf2image import convert_from_path
from PIL import Image, ImageOps 

# Get the input PDF path
pdf_path = os.path.join("Financials", "BS", "Apple_bs_21.pdf")

# Convert first page of PDF to image
images = convert_from_path(pdf_path, dpi=300)
pdf_basename = os.path.splitext(os.path.basename(pdf_path))[0]

higher_clarity = images[0].convert("L")

higher_clarity = ImageOps.autocontrast(higher_clarity)

scale_factor = 2  
larger = higher_clarity.resize(
    (higher_clarity.width * scale_factor, higher_clarity.height * scale_factor)
)

image_filename = os.path.splitext(os.path.basename(pdf_path))[0]
cache_file = os.path.join("Cache", f"ocr_cache_{image_filename}.pkl")

def remove_horizontal_lines(pil_image):
    # Convert PIL to OpenCV grayscale
    img = np.array(pil_image)

    # Binarize image (invert for easier line detection)
    _, binary = cv2.threshold(img, 160, 255, cv2.THRESH_BINARY_INV) # Very important: first number is threshold for whether something is treated as white or black which can delete entire sections if not careful

    # Detect horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    detected_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)

    # Visualize horizontal lines on original for debugging
    color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    color_img[detected_lines > 0] = [0, 255, 0]  # green for detected lines
    debug_line_path = os.path.join("Debug", "debug_detected_lines.png")

    Image.fromarray(color_img).save(debug_line_path)

    # Remove horizontal lines from binary
    no_lines = cv2.bitwise_not(binary)
    no_lines = cv2.bitwise_or(no_lines, detected_lines)
    cleaned = cv2.bitwise_not(no_lines)

    # Connected component analysis to remove small blobs
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    filtered = np.zeros_like(cleaned)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= 20:  # keep components with sufficient area
            filtered[labels == i] = 255

    # Convert to RGB for PIL
    cleaned_rgb = cv2.cvtColor(filtered, cv2.COLOR_GRAY2RGB)
    final_image = Image.fromarray(cleaned_rgb)

    return final_image, debug_line_path

def get_data(cache=True):

    if cache and os.path.exists(cache_file):
        #os.remove(cache_file) 
        with open(cache_file, "rb") as f:
            results = pickle.load(f)
        print(f"Loaded OCR results from cache file: {cache_file}")
        return results, processed_img
    else:
        reader = easyocr.Reader(['en'])
        processed, debug_path = remove_horizontal_lines(larger)
        print(f"Debug image saved to: {debug_path}")
        results = reader.readtext(np.array(processed), width_ths=0.5)
        with open(cache_file, "wb") as f:
            pickle.dump(results, f)
    
    return results, processed

def get_coords(bbox):
    tl, tr, _, _ = bbox
    x1, y = tl
    x2, _ = tr
    return (x1, x2, y)

def get_y_bounds(list_of_coords, num_years):
    cols = [[] for _ in range(num_years)]  # Properly allocate lists

    for coords in list_of_coords:
        if not len(coords) == num_years:
            continue
        for i, coord in enumerate(coords):
            cols[i].append(coord)

    col_coords = [np.median(col) for col in cols]
    return col_coords

def glue_malformed(malformed, lines, debug=False):
    for data in malformed:
        text, y = data
        #for line in lines:
           # if 
        

def build_fs(ocr_output, debug=False, y_thresh=100, val_x_thresh=25): # usually y_thresh 50
    num_vals = []
    all_val_coords = []
    lines = []
    detect_vals = re.compile(r'^\(?-?\$?\d{1,3}(?:,\d{3})*(?:\.\d+)?\)?$')

    start_bbox, first_line, _ = ocr_output[0]
    if debug:
        print(f'Got start line: {first_line}')
    _, _, start_y = get_coords(start_bbox)
    start_line = [first_line]

    line_val_coords = []
    for bbox, line, _ in ocr_output[1:]: # issue rn where it skips the last line.
        x1, x2, y = get_coords(bbox)
        if y in range(start_y - y_thresh, start_y + y_thresh):
            if debug:
                print(f'LINE: {line} on same line')
            if detect_vals.match(line):
                if debug:
                    print(f'DETECTED VAL: {line}')
                line_val_coords += [x2]
                start_line += [(line, x2)]
            else:
                if debug:
                    print(f'DETECTED TEXT {line}')
                if line == '$':
                    continue
                start_line += [line]
        else:
            if debug:
                print(f'NEW LINE DETECTED: {line}')
                print(f'APPENDING COMPLETED LINE {start_line}')
            lines.append(start_line)
            start_line = [line]
            start_y = y
            num_vals.append(len(line_val_coords))
            if not line_val_coords == []:
                all_val_coords.append(line_val_coords)
            line_val_coords = []
    if start_line:
        lines.append(start_line)
        if line_val_coords:
            all_val_coords.append(line_val_coords)
        num_vals.append(len(line_val_coords))

    num_years = Counter(num_vals).most_common(1)[0][0]
    if debug:
        print(f'NUM YEARS: {num_years}')
        print(f'VAL COORDS: {all_val_coords}')
    
    col_coords = get_y_bounds(all_val_coords, num_years)
    return (col_coords, lines)

def debug_output(data, processed_img, verbose=False):
    img = np.array(processed_img)

    for bbox, text, conf in data:
        if verbose:
            print(f"{text} (confidence: {conf:.2f})")
        bbox = [tuple(map(int, point)) for point in bbox]
        cv2.rectangle(img, bbox[0], bbox[2], (0, 255, 0), 2)
        cv2.putText(img, text, bbox[0], cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    cv2.imwrite(os.path.join("Debug", "debug_overlay.png"), img)

class LineItem:
    def __init__(self, name):
        assert isinstance(name, str)
        self.name = name
        self.data = {}

    def add_data(self, year, value=None):
        assert isinstance(year, int)
        assert isinstance(value, (int, float))
        self.data[year] = value

    def __str__(self):
        data_pairs = [(year, self.data[year]) for year in sorted(self.data.keys())]
        return f'Line item: {self.name} | Values: {data_pairs}'
    
class FinancialStatement:
    def __init__(self):
        self.lines = []

    def add_line_item(self, data):
        assert isinstance(data, (LineItem))
        self.lines.append(data)

    def __str__(self):
        return "\n".join(str(line) for line in self.lines)

processed_img = remove_horizontal_lines(larger)[0]
ocr_output, _ = get_data()
#for bbox, line, _ in ocr_output:
 #   print(bbox, line) 
col_coords, lines = build_fs(ocr_output, True)
print(f'===COL COORDS=== {col_coords}')
print()
print(f'===Lines=== {lines}')
#debug_output(ocr_output, processed_img)