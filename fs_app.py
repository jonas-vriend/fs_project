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

pdf_path = os.path.join("Financials", "BS", "UHG_bs_18.pdf")  # I change this to test different statements
detect_vals = re.compile(r'^\(?-?[$S]?\s?\d{1,3}(?:,\d{3})*(?:\.\d+)?\)?$')

################################################################################


def preprocess_img(debug):
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

    # ---- Summing lines (red) ---- #
    summing_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (70, 1))
    summing_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, summing_kernel, iterations=2)

    # ---- Underscore candidates (green) ---- #
    underscore_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (70, 1))
    raw_underscores = cv2.morphologyEx(binary, cv2.MORPH_OPEN, underscore_kernel, iterations=1)

    # ---- Isolate underscore lines NOT overlapping summing lines ---- #
    filtered_underscores = cv2.subtract(raw_underscores, summing_lines)

    # ---- Create debug overlay ---- #
    color_overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    color_overlay[summing_lines > 0] = [255, 0, 0]      # Red for summing lines
    color_overlay[filtered_underscores > 0] = [0, 255, 0]  # Green for underscore 0s

    # ---- Remove summing lines from image ---- #
    binary_no_summing = cv2.bitwise_not(binary)
    no_summing = cv2.bitwise_or(binary_no_summing, summing_lines)
    cleaned = cv2.bitwise_not(no_summing)

    # ---- Remove small noise blobs ---- #
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    filtered = np.zeros_like(cleaned)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= 20:
            filtered[labels == i] = 255

    final_image = Image.fromarray(cv2.cvtColor(filtered, cv2.COLOR_GRAY2RGB))

    # ---- Extract bounding boxes of underscore lines ---- #
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

    # ---- Save debug overlay ---- #
    debug_path = os.path.join("Debug", "debug_detected_lines.png")
    Image.fromarray(color_overlay).save(debug_path)

    return final_image, debug_path, underscore_coords



def get_data(cache=True, debug=False):
    """
    Loads or creates cached file
    """
    processed, debug_path, underscore_coords = preprocess_img(debug)
    image_filename = os.path.splitext(os.path.basename(pdf_path))[0]
    cache_file = os.path.join("Cache", f"ocr_cache_{image_filename}.pkl")

    if cache and os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            results = pickle.load(f)
        print(f"Loaded OCR results from cache file: {cache_file}")
        return results, processed, underscore_coords
    else:
        reader = easyocr.Reader(['en'])
        print(f"Debug image saved to: {debug_path}")
        results = reader.readtext(
            np.array(processed), 
            width_ths=0.5,
            text_threshold=0.55  # Very important: bringing this down from default allows for detection of single digit values 
        )
        with open(cache_file, "wb") as f:
            pickle.dump(results, f)
        return results, processed, underscore_coords


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

        if len(text) == 1 and text not in {'$', 'S'}:
            continue
        x1, x2, y = get_coords(bbox)
        if y in range(start_y - y_thresh, start_y + y_thresh):

            if detect_vals.match(text):
                if debug:
                    print(f'DETECTED VAL: {text}')
                if '$' in text or 'S' in text:
                    new_line.add_dollar_sign()
                    text = text.replace('S', '').replace('$', '')
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
            new_line.add_y_val(start_y)
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
            start_line = text

    # Handle last line
    if start_line:
        new_line.add_text(start_line)
        new_line.add_x_coords(current_x1, x2)
        new_line.add_y_val(start_y)
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

def add_underscore_zeros(lines, col_coords, underscore_coords, debug=False, val_x_thresh=75, y_thresh=60):
    can_num = 0
    for (x, y, w, _) in underscore_coords:
        can_num += 1
        if debug:
            print('\n=== NEW CANDIDATE ===')
        rx = x + w
        for line in lines:
            line_y = line.get_y_val()
            vals = line.get_vals()
            if abs(y - line_y) < y_thresh:

                if len(col_coords) > len(vals):
                    candidate_distances = [abs(rx - col_x) for col_x in col_coords]
                    candidate_closest_idx = np.argmin(candidate_distances)
                    candidate_place = col_coords[candidate_closest_idx]

                    if candidate_distances[candidate_closest_idx] <= val_x_thresh:
                        val_coords = [val_coord for _, val_coord in vals]
                        accounted_for = []

                        for val_coord in val_coords:
                            val_distances = [abs(val_coord - col_x) for col_x in col_coords]
                            val_closest_idx = np.argmin(val_distances)
                            accounted_for.append(col_coords[val_closest_idx])

                        if candidate_place in accounted_for:
                            if debug:
                                print(f'REJECTING candidate {can_num} with coords x: {rx} y: {y}. Val already exists in this position. LINE: {line.get_text()} LINE Y: {line_y}')
                            continue

                        else:
                            added = False
                            for i, coord in enumerate(accounted_for):
                                if candidate_place < coord:
                                    line.vals.insert(i, ('0', rx))
                                    added = True
                                    break
                            
                            if not added:
                                line.vals.append(('0', rx))
                            if debug:
                                print(f'INSERTED candidate {can_num} AT X: {rx}, Y: {y}, LINE: {line.get_text()} LINE_Y {line_y}')
                            break
                    else:
                        if debug:
                            print(f'REJECTING candidate {can_num} with coords x: {rx} y: {y}. Failed x threshold check. LINE: {line.get_text()}')
                        continue
                else:
                    if debug:
                        print(f'REJECTING candidate {can_num} with coords x: {rx} y: {y}. Vals already accounted for. LINE: {line.get_text()}')
                    continue
            else:
                if debug:
                    print(f'REJECTING candidate {can_num} with coords x: {rx} y: {y}. Y thresh failed. LINE: {line.get_text()} Y: {line_y}')     
    return

def add_indentation(raw_data, debug=False, x_thresh=50):
    if debug:
        print("Running add_indentation...")

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
            if debug:
                print(f"Case 1: indent += 1 -> {current_indent}")
        elif abs(x1 - min_x) < x_thresh:
            current_indent = 0
            current_baseline = min_x
            line.add_indentation(current_indent)
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
            if debug:
                print(f"Case 4: indent -= 1 -> {current_indent}")
        else:
            if debug:
                print(f"NO CASE MATCHED FOR LINE: {line}")


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

def merge_lines(lines, debug=False):
    """
    Merges lines that are split into two lines IF:
    - The second line does not start with a capitalized letter
    - The second line is indented farther than the first
    - Either line has values but not both
    """
    output = []

    for line in lines:
        label, _, vals, d_sign, indent, _ = line.get_all()

        if not label:
            output.append(line)
            continue

        first_char = label[0]

        if not (first_char.isalpha() and first_char.isupper()):
            if not output:
                output.append(line)
                continue
            prev = output[-1]
            prev_label, _, prev_vals, _, prev_indent, _ = prev.get_all()
            
            if prev_indent < indent:
                if (vals and prev_vals) or (not vals and not prev_vals):
                    if debug:
                        print(f'Tried to merge but ran into Error. At least one line must have vals but not both. Lines: {label} | {prev_label}')
                    output.append(line)
                    continue
                full_label = prev_label.strip() + ' ' + label.strip()
                if debug:
                    print(f'MERGIING LINE. LINE NOW: {full_label}')
                if vals:
                    prev.vals = vals
                if d_sign:
                    prev.add_dollar_sign()
                prev.add_text(full_label)
            else:
                if debug:
                    print(f'Warning. Line: {label} is not capitalized but could not merge')
                output.append(line)
        else:
            output.append(line)

    return output


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
    years = ['Year ' + str(num + 1) for num in range(len(col_coords))] # Default years in case years not found

    for line in lines:
        label = line.get_text()
        label = label.strip('.').strip('_')  # for formats where labels and vals separated with periods
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

            for val, val_coord in vals:
                # Find closest column index
                distances = [abs(val_coord - col_x) for col_x in col_coords]
                closest_idx = np.argmin(distances)

                if distances[closest_idx] <= val_x_thresh:
                    if detect_vals.match(val):
                        cleaned = val.replace(',', '').replace('_', '').replace('(', '-').replace(')', '')
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
                        label = label + ' ' + str(val) 
                        line.add_text(label)
                        if debug:
                            print(f'ADDING {val} TO END OF LABEL. LABEL NOW: {label}')
                    elif debug:
                        print(f'REJECTED {val} AT X = {val_coord}, TOO FAR FROM LABEL {lab_x2}')

            new_line_item.add_label(label)
            new_fs.add_line_item(new_line_item)

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
        self.y_val = None
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

    def add_indentation(self, val):
        self.indent = val

    def add_y_val(self, y):
        self.y_val = y

    def get_text(self):
        return self.text

    def get_x_coords(self):
        return self.text_x_coords

    def get_vals(self):
        return self.vals

    def get_dollar_sign(self):
        return self.dollar_sign

    def get_indent(self):
        return self.indent
    
    def get_y_val(self):
        return self.y_val

    def get_all(self):
        return self.text, self.text_x_coords, self.vals, self.dollar_sign, self.indent, self.y_val

    def __str__(self):
        return f"TEXT: {self.text} | COORDS: {self.text_x_coords} | VALS: {self.vals}"


class LineItem:
    def __init__(self):
        self.label = None
        self.dollar_sign = False
        self.indent = 0
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
    ocr_output, processed_img, underscore_coords = get_data(use_cache, debug)
    col_coords, lines = preprocess_text(ocr_output, debug)
    if underscore_coords:
        add_underscore_zeros(lines, col_coords, underscore_coords, debug)
    debug_output(ocr_output, processed_img, col_coords)
    add_indentation(lines)
    merged_lines = merge_lines(lines, debug)
    completed = build_fs(col_coords, merged_lines, debug)
    export_fs(completed, export_filename)


if __name__ == "__main__":
    main(debug=True, use_cache=True)