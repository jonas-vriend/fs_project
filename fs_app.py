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


########################### GLOBAL VARIABLES ###################################

pdf_path = os.path.join("Financials", "IS", "UHG_is_20.pdf") # I  change this to test different statements
detect_vals = re.compile(r'^\(?-?[$S]?\s?\d{1,3}(?:,\d{3})*(?:\.\d+)?\)?$')

################################################################################


def preprocess_img():
    """
    Converts PDF to image, increases clarity, inverts to black background and white text to improve OCR performance, removes summing lines.
    """
    image = convert_from_path(pdf_path, dpi=300)

    higher_clarity = image[0].convert("L")

    higher_clarity = ImageOps.autocontrast(higher_clarity)

    scale_factor = 2  
    larger = higher_clarity.resize(
        (higher_clarity.width * scale_factor, higher_clarity.height * scale_factor)
    )

    # Convert PIL to OpenCV grayscale
    img = np.array(larger)

    # Binarize image (invert for easier line detection)
    _, binary = cv2.threshold(img, 160, 255, cv2.THRESH_BINARY_INV) # Very important: first number arg is threshold for whether something is treated as white or black which can delete entire sections if not careful. Consider making this dynamic in the future

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

    # Connected component analysis to remove small white artifacts
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
    """
    Loads or creates cached file
    """
    processed, debug_path = preprocess_img()
    image_filename = os.path.splitext(os.path.basename(pdf_path))[0]
    cache_file = os.path.join("Cache", f"ocr_cache_{image_filename}.pkl")

    if cache and os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            results = pickle.load(f)
        print(f"Loaded OCR results from cache file: {cache_file}")
        return results, processed
    else:
        reader = easyocr.Reader(['en'])
        print(f"Debug image saved to: {debug_path}")
        results = reader.readtext(np.array(processed), width_ths=0.5)
        with open(cache_file, "wb") as f:
            pickle.dump(results, f)
        return results, processed


def get_coords(bbox):
    """
    Unpacks bbox coordinates from OCR results 
    """
    tl, tr, _, _ = bbox
    x1, y = tl
    x2, _ = tr
    return (x1, x2, y)


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


def preprocess_text(ocr_output, debug=False, y_thresh=30):
    """
    Builds RawData objects for each line. Also determines position of line item values by calling get_x_bounds()
    """
    num_vals = [] # stores number of vals to determine how many columns to expect
    all_val_coords = [] # stores val coordinates to determine where vals should be on page
    output = [] # what is returned
    line_val_coords = []  # Holds x2 coords of values on current line to get val boundaries

    #Starts very first line for loop to compare to
    start_bbox, first_line, _ = ocr_output[0]
    x1, x2, start_y = get_coords(start_bbox)
    current_x1 = x1
    start_line = first_line

    new_line = RawData()
    for bbox, text, _ in ocr_output[1:]:
        x1, x2, y = get_coords(bbox)

        if y in range(start_y - y_thresh, start_y + y_thresh):

            if detect_vals.match(text):
                if debug:
                    print(f'DETECTED VAL: {text}')
                line_val_coords.append(x2)
                new_line.add_val(text, x2)

            else:
                if text.strip() in {'$', 'S'}:
                    new_line.add_dollar_sign()
                    continue  # $ added in excel formatting. Do not add to RawData
                else:
                    if debug:
                        print(f'DETECTED LABEL FRAGMENT: {text}')
                    start_line += ' ' + text

        else:
            # Finalize previous line
            new_line.add_text(start_line)
            new_line.add_x_coords(current_x1, x2)
            if debug:
                print(f'COMPLETED LINE: {start_line}, VALS: {new_line.get_vals()}')
            if line_val_coords: # good dont need to change
                all_val_coords.append(line_val_coords)
                num_vals.append(len(line_val_coords))
            output.append(new_line)

            # Reset everything for new line
            new_line = RawData()
            start_y = y
            line_val_coords = []
            current_x1 = x1

            # First fragment of new line
            if detect_vals.match(text):
                line_val_coords.append(x2)
                new_line.add_val(text, x2)
            elif text.strip() != '$':
                start_line = text

    # Handle last line
    if start_line:
        new_line.add_text(start_line)
        new_line.add_x_coords(current_x1, x2)
        if debug:
            print(f'ADDING FINAL LINE: {start_line}')
        if line_val_coords:
            all_val_coords.append(line_val_coords)
            num_vals.append(len(line_val_coords))
    output.append(new_line)

    num_years = Counter(num_vals).most_common(1)[0][0]
    if debug:
        print(f'NUM YEARS: {num_years}')
        print(f'VAL COORDS: {all_val_coords}')
    
    col_coords = get_x_bounds(all_val_coords, num_years)
    if debug:
        print(f'Captured col_coords: {col_coords}')
    return col_coords, output

def add_indentation(raw_data, debug=False, x_thresh=75):
    if debug:
        print("Running add_indentation...")

    output = []

    # initialize min x
    x1, _ = raw_data[0].get_x_coords()
    min_x = x1

    for line in raw_data[1:]:
        x1, _ = line.get_x_coords()
        if x1 < min_x:
            min_x = x1

    current_indent = 0
    current_baseline = min_x
    for line in raw_data:
        x1, _ = line.get_x_coords()
        if debug:
            print(f"\nProcessing {line.get_text()}")
            print(f"x1: {x1}, baseline: {current_baseline}, min_x: {min_x}")
        
        if x1 - current_baseline > 0 and abs(x1 - current_baseline) > x_thresh:
            current_indent += 1
            current_baseline = x1
            line.add_indentation(current_indent)
            output.append(line)
            if debug:
                print(f"Case 1: indent += 1 -> {current_indent}")
        elif abs(x1 - min_x) < x_thresh:
            current_indent = 0
            current_baseline = min_x
            line.add_indentation(current_indent)
            output.append(line)
            if debug:
                print(f"Case 2: indent = 0")
        elif abs(x1 - current_baseline) < x_thresh:
            current_baseline = x1
            line.add_indentation(current_indent)
            if debug:
                print(f"Case 3: within threshold, indent stays {current_indent}")
        elif x1 - current_baseline < 0 and abs(x1 - current_baseline) > x_thresh:
            current_indent -= 1
            current_baseline = x1
            line.add_indentation(current_indent)
            output.append(line)
            if debug:
                print(f"Case 4: indent -= 1 -> {current_indent}")
        else:
            if debug:
                print(f"NO CASE MATCHED FOR LINE: {line}")

    return output

def what_fs(cleaned):
    """
    - Iterates through labels of RawData objects to determine the type of financial statement.
    - Checks if expected words are in labels. 
    - If a word is found, adds 1 to the score associated with that statement (ie if BS word is found adds +1 to BS score).
    - The statement type will be determined based on whether more IS words or BS words are detected
    """
    BALANCE_SHEET_TERMS = ['balance sheet', 'asset', 'assets', 'liability',
                            'liabilities', 'inventory', 'inventories', 'property', 'plant', 'equipment',
                            'accounts payable', 'deferred revenue', "shareholder's equity", 'common stock', 'accounts receivable',
                            'additional paid-in capital', 'accumulated other comprehensive income', 'cash and cash equivalents', 'retained earnings']
    
    INCOME_STATEMENT_TERMS = ['income statement', 'revenue', 'sales', 'gross margin', 'operating expenses', 'cost of goods sold',
                              'research and development', 'net income', 'income', 'depreciation', 'tax', 'taxes']
    
    bs_score = 0
    is_score = 0

    for line in cleaned:
        text = line.get_text()
        for b_term in BALANCE_SHEET_TERMS:
            if b_term in text.lower():
                bs_score += 1
        
        for i_term in INCOME_STATEMENT_TERMS:
            if i_term in text.lower():
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
    """
    Determines Regex to use to identify end of statement
    """
    if result == 'BALANCE_SHEET':
        return re.compile(r'(?i)total.*liabilit(?:y|ies).*equity')
    else:
        return re.compile(r'(?i)^.*net\s+\(?(income|earnings)\)?(?:\s+\(loss\))?.*')
    

def build_fs(col_coords, lines, debug=False, val_x_thresh=75):
    """
    Builds FinancialStatement object
    """
    extract_date = re.compile(r'^(?:\D*?(?:19|20)\d{2}){2,}\D*$')
    year_pattern = re.compile(r'(?:19|20)\d{2}')

    got_years = False
    end_collection = False
    new_fs = FinancialStatement()
    fs_type = what_fs(lines)
    end = get_end(fs_type)
    years = ['Year ' + str(num + 1) for num in range(len(col_coords))] # Default years in case yeaars not found

    for line in lines:
        label = line.get_text()
        label = label.replace('_', '').replace('.', '')
        dollar_sign = line.get_dollar_sign()

        if not got_years and extract_date.search(label):
                matches = year_pattern.findall(label)
                years = [int(y) for y in matches]
                got_years = True
                new_fs = FinancialStatement()
                if debug:
                    print('CAPTURED YEARS:', years)
                continue

        elif end_collection and not end.match(label):
            if debug:
                print(f'Final line detected. Excluded: {label}')
            break

        new_line_item = LineItem()

        if dollar_sign:
            new_line_item.add_dollar_sign()

        vals = line.get_vals()
        if not line.get_vals() == []:
            if end.match(label) and dollar_sign:
                if debug:
                    print(f'Final line detected and included: {label}')
                end_collection = True

            if debug:
                print(f'LINE ITEM DETECTED: {label}')

            assigned_vals = [None] * len(col_coords)

            skip_erroneous_val = False
            for val, val_coord in vals:
                # Find closest column index
                distances = [abs(val_coord - col_x) for col_x in col_coords]
                closest_idx = np.argmin(distances)

                if distances[closest_idx] <= val_x_thresh:
                    if detect_vals.match(val):
                        cleaned = val.replace('$', '').replace(',', '').replace('_', '').replace('(', '-').replace(')', '').replace('S', '')
                        try:
                            val_num = int(cleaned)
                        except ValueError:
                            val_num = 0
                            if debug:
                                print(f'COULD NOT PARSE VAL. TREATING AS 0: {val}')
                    else:
                        val_num = 0
                        if debug:
                            print(f'VAL NOT RECOGNIZED. TREATING AS 0: {val}')
                    assigned_vals[closest_idx] = val_num
                else:
                    if debug:
                        print(f'REJECTED: {val} AT X = {val_coord}, TOO FAR FROM COL {col_coords[closest_idx]}')
                    _, lab_x2 = line.get_x_coords()
                    if abs(val_coord - lab_x2) <= 600:
                        label = label + ' ' +  str(val) 
                        if debug:
                            print(f'ADDING {val} TO END OF LABEL. LABEL NOW: {label}')
                    elif debug:
                        print(f'REJECTED {val} AT X = {val_coord}, TOO FAR FROM LABEL {lab_x2}')
                    skip_erroneous_val = True

            new_line_item.add_label(label)
            new_fs.add_line_item(new_line_item)
    
            if skip_erroneous_val: # Need to add an additional check shouldnt just concatenate without checking coords first. 
                continue

            # Fill missing years with 0
            for idx, year in enumerate(years):
                value = assigned_vals[idx] if assigned_vals[idx] is not None else 0
                new_line_item.add_data(year, value)

        else:
            if end_collection:
                if debug:
                    print(f'END REACHED. EXCLUDED: {label}')
                break

            if debug:
                print(f'HEADING DETECTED: {label}')
            new_line_item.add_label(label)
            new_fs.add_line_item(new_line_item)
    if not got_years and debug:
        print('WARNING COULD NOT FIND YEARS.')
    return new_fs


def export_fs(fs, filename):
    """
    Exports FinancialStatement object to CSV
    """
    all_years = []
    seen = set()
    for item in fs.lines:
        for year in item.data.keys():
            if year not in seen:
                seen.add(year)
                all_years.append(year)

    with open(filename, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Label"] + all_years)

        for item in fs.lines:
            row = [item.label]
            for year in all_years:
                value = item.data.get(year, "")
                row.append(value)
            writer.writerow(row)


def debug_output(data, processed_img, col_coords, val_x_thresh=75, debug=False):
    """
    - Draws bboxes and red lines denoting value position on financial statement.
    - Prints text captured in each bbox and its confidence value
    """
    img = np.array(processed_img)

    for bbox, text, conf in data:
        if debug:
            print(f"{text} (confidence: {conf:.2f})")
        bbox = [tuple(map(int, point)) for point in bbox]
        cv2.rectangle(img, bbox[0], bbox[2], (0, 255, 0), 2)
        cv2.putText(img, text, bbox[0], cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    if col_coords:
        height = img.shape[0]
        for x in col_coords:
            x_int = int(round(x))
            # Vertical dashed red line
            for y in range(0, height, 20):
                cv2.line(img, (x_int, y), (x_int, y + 10), (0, 0, 255), 5)

            # Confidence bar (horizontal range = ±val_x_thresh)
            err_bar_y = int(height * 0.50)
            left = int(x - val_x_thresh)
            right = int(x + val_x_thresh)
            cv2.line(img, (left, err_bar_y), (right, err_bar_y), (0, 0, 255), 5)

    cv2.imwrite(os.path.join("Debug", "debug_overlay.png"), img)


class RawData:
    def __init__(self):
        self.text = None
        self.text_x_coords = (None, None)
        self.dollar_sign = False
        self.indent = 0
        self.vals = []  # List of (val, x_coord) pairs

    def add_text(self, text):
        assert isinstance(text, str), "Text must be a string"
        self.text = text

    def add_x_coords(self, coord1, coord2):
        self.text_x_coords = (float(coord1), float(coord2))

    def add_val(self, val, coord):
        self.vals.append((val, float(coord)))

    def add_dollar_sign(self):
        self.dollar_sign = True

    def get_text(self):
        return self.text

    def get_x_coords(self):
        return self.text_x_coords

    def get_vals(self):
        return self.vals
    
    def get_dollar_sign(self):
        return self.dollar_sign
    
    def add_indentation(self, val):
        self.indent = val

    def __str__(self):
        return f"TEXT: {self.text} | COORDS: {self.text_x_coords} | VALS: {self.vals}"


class LineItem:
    def __init__(self):
        self.label = None
        self.dollar_sign = False
        self.data = {}

    def add_label(self, label):
        assert isinstance(label, str)
        self.label = label

    def add_data(self, year, value=None):
        assert isinstance(value, (int, float))
        self.data[year] = value

    def add_dollar_sign(self):
        self.dollar_sign = True

    def get_data(self):
        return self.data
    
    def get_dollar_sign(self):
        return self.dollar_sign

    def __str__(self):
        data_pairs = [(year, self.data[year]) for year in sorted(self.data.keys())]
        return f'Line item: {self.label} | Values: {data_pairs}'


class FinancialStatement:
    def __init__(self):
        self.lines = []

    def add_line_item(self, data):
        assert isinstance(data, (LineItem))
        self.lines.append(data)
    
    def get_lines(self):
        return self.lines

    def __str__(self):
        return "\n".join(str(line) for line in self.lines)


def main(debug=False, use_cache=False, export_filename="financial_statement.csv"):
    """
    Orchestrates the pipeline:
    - Load/caches OCR
    - Extracts structure and values
    - Builds financial statement object
    - Exports as CSV
    """
    print("MAIN FUNCTION DEBUG VALUE:", debug)
    ocr_output, processed_img = get_data(cache=use_cache)
    col_coords, lines = preprocess_text(ocr_output, debug)
    debug_output(ocr_output, processed_img, col_coords)
    print("DEBUG INSIDE add_indentation:", debug)
    indented_lines = add_indentation(lines, debug)
    completed = build_fs(col_coords, indented_lines, debug)
    export_fs(completed, export_filename)


if __name__ == "__main__":
    main(debug=True, use_cache=True)
