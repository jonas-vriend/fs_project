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

## GLOBAL VARIABLES ##

detect_vals = re.compile(r'^\(?-?\$?\d{1,3}(?:,\d{3})*(?:\.\d+)?\)?$')

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
        

def preprocess_text(ocr_output, debug=False, y_thresh=50):
    num_vals = []
    all_val_coords = []
    lines = []

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

def what_fs(cleaned):
    BALANCE_SHEET_TERMS = ['balance sheet', 'asset', 'assets', 'liability',
                            'liabilities', 'inventory', 'inventories', 'property', 'plant', 'equipment',
                            'accounts payable', 'deferred revenue', "shareholder's equity", 'common stock', 'accounts receivable',
                            'additional paid-in capital', 'accumulated other comprehensive income', 'cash and cash equivalents', 'retained earnings']
    
    INCOME_STATEMENT_TERMS = ['income statement', 'revenue', 'sales', 'gross margin', 'operating expenses', 'cost of goods sold',
                              'research and development', 'net income', 'income', 'depreciation', 'tax', 'taxes']
    
    bs_score = 0
    is_score = 0

    for line, _ in cleaned:
        for b_term in BALANCE_SHEET_TERMS:
            if b_term in line:
                bs_score += 1
        
        for i_term in INCOME_STATEMENT_TERMS:
            if i_term in line:
                is_score += 1
    
    if bs_score > is_score and bs_score >= 5:
        print(f'BS identified. BS score: {bs_score} IS score: {is_score}')
        return 'BALANCE_SHEET'
    elif is_score > bs_score and is_score >= 5:
        print(f'IS identified. IS score: {is_score} BS score: {bs_score}')
        return 'INCOME_STATEMENT'
    else:
        raise ValueError('Could not recognize document')


def get_end(result):
    if result == 'BALANCE_SHEET':
        return re.compile(r'(?i)total.*liabilit(?:y|ies).*equity.*\d+')
    else:
        return re.compile(r'(?i)^.*net\s+\(?income\)?(?:\s+\(loss\))?.*')


def build_fs(col_coords, lines, debug=False, val_x_thresh=25):
    extract_date = re.compile(r'^(?:\D*?(?:19|20)\d{2}){2,}\D*$')
    year_pattern = re.compile(r'(?:19|20)\d{2}')

    got_years = False #  flag that tells computer to skip useless lines until dates found
    end_collection = False #  flag that tells computer to exclude useless lines after final line item found
    new_fs = FinancialStatement()
    fs_type = what_fs(lines)
    end = get_end(fs_type) #  regex computer uses to determine end

    for line in lines:
        if not got_years:
            found_years = extract_date.search(line)
            if found_years:
                matches = year_pattern.findall(line)
                years = [int(y) for y in matches]
                got_years = True
                if debug:
                    print('CAPTURED YEARS:', years)
                continue
            else:
                if debug:
                    print('SKIPPED LINE BEFORE YEARS CAPTURED:', line)
                continue

        label = line[0]

        if end.match(line):
                if debug:
                    print(f'Final line detected and included: {line}')
                end_collection = True
        elif end_collection:
            if debug:
                print(f'Final line detected. Excluded: {line}')
            break

        new_line_item = LineItem(label)

        if len(line) > 1:

            vals = line[1:]
            if debug:
                print(f'LINE ITEM DETECTED: {label}')

            for year, col_coord in zip(years, col_coords):
                for val, val_coord in vals:
                    if val_coord in range(col_coord - val_x_thresh, col_coord + val_x_thresh):
                        stripped_val = val.replace('$', '').replace('_', '').replace(',', '').replace('(', '-').replace(')', '')
                        if detect_vals.match(stripped_val):
                            if debug:
                                print(f'NUMERIC VAL DETECTED: {val}')
                            new_line_item.add_data(year, stripped_val)
                        else:
                            if debug:
                                print(f'NUMERIC VAL NOT DETECTED. TREATING AS 0: {val}')
                            new_line_item.add_data(year, 0)
                    else:
                        if debug:
                            print('LINE FAILED COORD CHECK: {label}')
                        continue
                if len(vals) < len(years) and year not in new_line_item.get_data().keys():
                    if debug:
                        print(f'NO VAL DETECTED. TREATING AS 0: {val}')
                        new_line_item.add_data(year, 0)
                    if end_collection:
                        if debug:
                            print(f'Final line detected. Excluded: {line}')
                        break
            
            
            new_fs.add_line_item(new_line_item)

        else:
            if end_collection:
                if debug:
                    print(f'END REACHED. EXCLUDED: {label}')
                break

            if debug:
                print(f'HEADING DETECTED: {label}')
            new_fs.add_line_item(new_line_item)

    return new_fs



                






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

    def get_data(self):
        return self.data

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
ocr_output, _ = get_data(False)
#for bbox, line, _ in ocr_output:
 #   print(bbox, line) 
col_coords, lines = preprocess_text(ocr_output, True)
print(f'===COL COORDS=== {col_coords}')
print()
print(f'===Lines=== {lines}')
#debug_output(ocr_output, processed_img)