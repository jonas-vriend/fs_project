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

pdf_path = os.path.join("Financials", "IS", "Walmart_is_24.pdf")  # I change this to test different statements
detect_vals = re.compile(r'^\(?-?[$S]?\s?\d{1,3}(?:,\d{3})*(?:\.\d+)?\)?$')  # Regex to detect financial values on right side of financial statement

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

    # Summing lines (red)
    summing_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (70, 1))
    summing_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, summing_kernel, iterations=2)

    # Underscore candidates (green)
    underscore_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (70, 1))
    raw_underscores = cv2.morphologyEx(binary, cv2.MORPH_OPEN, underscore_kernel, iterations=1)

    # Isolate underscore lines NOT overlapping summing lines
    filtered_underscores = cv2.subtract(raw_underscores, summing_lines)

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

    return final_image, debug_path, underscore_coords


def get_data(cache=True, debug=False):
    """
    Runs the OCR pipeline on the current PDF.

    If a cached OCR result exists for the PDF, it is loaded from disk to save time.
    Otherwise, the image is processed, OCR is run using EasyOCR, and the results are cached.

    Returns:
        - results: the OCR output (list of [bbox, text, confidence] from EasyOCR)
        - processed: the final image after preprocessing (PIL.Image)
        - underscore_coords: list of bounding boxes for detected underscore "0" lines
    """
    # Preprocess the PDF and detect underscore lines and debug overlay path
    processed, debug_path, underscore_coords = preprocess_img(debug)

    # Create filename for cache 
    image_filename = os.path.splitext(os.path.basename(pdf_path))[0]
    cache_file = os.path.join("Cache", f"ocr_cache_{image_filename}.pkl")

    # If cache exists and allowed, load results from disk
    if cache and os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            results = pickle.load(f)
        print(f"Loaded OCR results from cache file: {cache_file}")
        return results, processed, underscore_coords
    
    # Otherwise, run OCR from scratch
    else:
        reader = easyocr.Reader(['en'])
        print(f"Debug image saved to: {debug_path}")
        results = reader.readtext(
            np.array(processed), 
            width_ths=0.5,  # Controls how aggressively EasyOCR merges text into the same box (0.5 results in many boxes)
            text_threshold=0.55   # Allows text with lower score to be captured. Lowering this improves recongition of single character values but also makes OCR hallucinate more
        )

        # Save OCR results to cache for future runs
        with open(cache_file, "wb") as f:
            pickle.dump(results, f)
        return results, processed, underscore_coords


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


def preprocess_text(ocr_output, debug=False, y_thresh=50):
    """
    Converts raw OCR output into structured line objects (RawData), identifying labels and numeric values.

    Args:
        ocr_output (list): List of OCR results, where each element is [bbox, text, confidence] from EasyOCR.
        debug (bool): If True, print detailed processing steps.
        y_thresh (int): Vertical threshold to determine if two OCR elements are on the same line.

    Returns:
        col_coords (list of float): Estimated x-coordinate midpoints for each financial column (one per year).
        output (list of RawData): Structured representation of each line on the financial statement.
    """

    num_vals = []  #Stores number of vals to determine how many columns to expect
    all_val_coords = []  # Stores val coordinates to determine where vals should be on page
    line_val_coords = []  # Holds x2 coords of values on current line to get val boundaries
    output = []  # What is returned - Final list of RawData objects

    # Initialize tracking variables with first OCR line
    start_bbox, first_line, _ = ocr_output[0]
    x1, x2, start_y1, start_y2 = get_coords(start_bbox)
    current_x1 = x1
    start_line = first_line
    new_line = RawData()

    # Process each OCR-detected segment starting from the second entry
    for bbox, text, _ in ocr_output[1:]:

        # Skip lone characters that are likely noise (except $ or S)
        if len(text) == 1 and text.isalpha() and text not in {'$', 'S'}:
            if debug:
                print(f'skipping noise: {text}')
            continue
        
        x1, x2, y1, y2 = get_coords(bbox)

        # Uses vertical allignment to determine whether to append current line or start a new one
        if abs(y1 - start_y1) < y_thresh:
            cleaned_text = (text.replace('S', '').replace('$', '')).strip()  # Meant to handle instances where $ of Y2 gets captured with value of Y1 ie '2,019 $'

            # Try to detect if the text is a financial value
            if detect_vals.match(cleaned_text):
                if debug:
                    print(f'DETECTED VAL: {cleaned_text}')

                # Vals with $ are tracked for openpyxl formatting
                if '$' in text or 'S' in text: 
                    if not text.startswith('$') or not text.startswith('S'):  # Prevents $ glued to end of val '2,019 $' from failing horizontal allignment check in build_fs()
                        DOLLAR_SIGN_CORRECTION = 50
                        x2 -= DOLLAR_SIGN_CORRECTION
                    new_line.add_dollar_sign()
            
                # Add value and its right x-bound
                line_val_coords.append(x2)
                new_line.add_val(cleaned_text, x2)

            else:
                # $ added in excel formatting. Do not add to RawData
                if text.strip() in {'$', 'S'}:
                    new_line.add_dollar_sign()
                    continue
                
                else:
                    # Glue OCR text fragments on same line together 
                    if debug:
                        print(f'DETECTED LABEL FRAGMENT: {text}')
                    start_line += ' ' + text

        else:
            # Finalize current line
            new_line.add_text(start_line)
            new_line.add_x_coords(current_x1, x2)
            new_line.add_y_vals(start_y1, start_y2)
            if debug:
                print(f'COMPLETED LINE: {start_line}, VALS: {new_line.get_vals()}')

            # Store value coordinate info for column estimation
            if line_val_coords:
                all_val_coords.append(line_val_coords)
                num_vals.append(len(line_val_coords))
            output.append(new_line)

            # Reset everything for new line
            new_line = RawData()
            start_y1 = y1
            start_y2 = y2
            line_val_coords = []
            current_x1 = x1
            start_line = text

    # Handle last line
    if start_line:
        new_line.add_text(start_line)
        new_line.add_x_coords(current_x1, x2)
        new_line.add_y_vals(start_y1, start_y2)
        if debug:
            print(f'ADDING FINAL LINE: {start_line}')
        if line_val_coords:
            all_val_coords.append(line_val_coords)
            num_vals.append(len(line_val_coords))
    output.append(new_line)

    # Infer number of years (columns) based on most common number of values per line
    num_years = Counter(num_vals).most_common(1)[0][0]
    if debug:
        print(f'NUM YEARS: {num_years}')
        print(f'VAL COORDS: {all_val_coords}')
    
    # Infer column boundaries based on median x position
    col_coords = get_x_bounds(all_val_coords, num_years)
    if debug:
        print(f'Captured col_coords: {col_coords}')
    return col_coords, output


def add_underscore_zeros(lines, col_coords, underscore_coords, debug=False, val_x_thresh=75):
    """
    Inserts '0' values into financial lines where underscores (e.g., "__") were detected in the image.

    Args:
        lines (list of RawData): Parsed financial lines with text and value positions.
        col_coords (list of float): Estimated x-coordinate centers for each year/column.
        underscore_coords (list of tuples): Coordinates of detected underscore lines (x, y, width, height).
        debug (bool): If True, prints detailed reasoning for accepting/rejecting candidates.
        val_x_thresh (int): Max horizontal distance allowed between underscore and column center to count as a match.
    """
    can_num = 0  # Candidate counter for debugging/tracing

    for (x, y, w, _) in underscore_coords:
        can_num += 1
        if debug:
            print('\n=== NEW CANDIDATE ===')

        rx = x + w  # Right x-coordinate of underscore (used to match against col_coords)

        for line in lines:
            line_y1, line_y2  = line.get_y_vals()
            vals = line.get_vals()

            # Check if underscore y-position falls within this line’s vertical bounds
            if y in range(line_y1, line_y2):

                # Only add a zero if the number of values on the line is less than number of columns
                if len(col_coords) > len(vals):

                    # Find closest column for underscore
                    candidate_distances = [abs(rx - col_x) for col_x in col_coords]
                    candidate_closest_idx = np.argmin(candidate_distances)
                    
                    # Check if the match is within the allowed horizontal threshold
                    if candidate_distances[candidate_closest_idx] <= val_x_thresh:

                        # Get existing column indices for real values in this line
                        existing_col_idxs = [
                            np.argmin([abs(val_coord - col_x) for col_x in col_coords])
                            for _, val_coord in vals
                        ]

                        # Check if that column is already filled
                        if candidate_closest_idx in existing_col_idxs:
                            if debug:
                                print(f'SKIPPING candidate {can_num}: column {candidate_closest_idx} already filled. LINE: {line.get_text()}')
                            continue

                        # Safe to insert - add a '0' value for the underscore
                        line.vals.append(('0', rx))
                        if debug:
                            print(f'INSERTED candidate {can_num} at col {candidate_closest_idx}, x={rx}, y={y}, LINE: {line.get_text()}')
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
                    print(f'REJECTING candidate {can_num} with coords x: {rx} y: {y}. Y thresh failed. LINE: {line.get_text()} LINE Y1: {line_y1} LINE Y2: {line_y2}')


def add_indentation(raw_data, debug=False, x_thresh=50):
    """
    Assigns indentation levels to each RawData line based on horizontal x-coordinates.

    Args:
        raw_data (list of RawData): Parsed lines from OCR with bounding box coordinates.
        debug (bool): If True, prints detailed info for debugging.
        x_thresh (int): Threshold for how much x1 must differ to count as a new indent level.
    """

    # Find the smallest x1 as the minimum indentation baseline
    x1, _ = raw_data[0].get_x_coords()
    min_x = x1
    for line in raw_data[1:]:
        x1, _ = line.get_x_coords()
        if x1 < min_x:
            min_x = x1

    # Initialize indentation tracking
    current_indent = 0
    current_baseline = min_x

    # Assign indentaion level to each line
    for line in raw_data:
        x1, _ = line.get_x_coords()
        if debug:
            print(f"\nProcessing {line.get_text()}")
            print(f"x1: {x1}, baseline: {current_baseline}, min_x: {min_x}")

        # Case 1: Line is further right than baseline and exceeds threhold so indent += 1
        if x1 - current_baseline > 0 and abs(x1 - current_baseline) > x_thresh:
            current_indent += 1
            current_baseline = x1
            line.add_indentation(current_indent)
            if debug:
                print(f"Case 1: indent += 1 -> {current_indent}")

        # Case 2: Line is roughly aligned with the leftmost margin so reset indent to 0
        elif abs(x1 - min_x) < x_thresh:
            current_indent = 0
            current_baseline = min_x
            line.add_indentation(current_indent)
            if debug:
                print(f"Case 2: indent = 0")
        
        # Case 3: Line is close to current baseline so maintain current indent
        elif abs(x1 - current_baseline) < x_thresh:
            current_baseline = x1
            line.add_indentation(current_indent)
            if debug:
                print(f"Case 3: within threshold, indent stays {current_indent}")

        # Case 4: Line is to the left of current indent baseline and exceeds threshold so indent -= 1
        elif x1 - current_baseline < 0 and abs(x1 - current_baseline) > x_thresh:
            current_indent -= 1
            current_baseline = x1
            line.add_indentation(current_indent)
            if debug:
                print(f"Case 4: indent -= 1 -> {current_indent}")

        # Warning print for edge cases
        else:
            if debug:
                print(f"NO CASE MATCHED FOR LINE: {line}")


def what_fs(cleaned, score_thresh=5):
    """
    Identifies the type of financial statement (Balance Sheet or Income Statement) from parsed text lines.
    - Iterates through labels of RawData objects to determine the type of financial statement.
    - Checks if expected words are in labels.
    - If a word is found, adds 1 to the score associated with that statement (ie if BS word is found adds +1 to BS score).
    - The statement type will be determined based on whether more IS words or BS words are detected

    Args:
        cleaned (list of RawData): A list of parsed text lines from the document, each with `.get_text()`.
        score_thresh (int): Threshold for whther the statement is identified or to raise an error.

    Returns:
        str: Either 'BALANCE_SHEET' or 'INCOME_STATEMENT' based on keyword frequency.
    """
    BALANCE_SHEET_TERMS = ['balance sheet', 'asset', 'assets', 'liability',
                            'liabilities', 'inventory', 'inventories', 'property', 'plant', 'equipment',
                            'accounts payable', 'deferred revenue', "shareholder's equity", 'common stock', 'accounts receivable',
                            'additional paid-in capital', 'accumulated other comprehensive income', 'cash and cash equivalents', 'retained earnings']
    
    INCOME_STATEMENT_TERMS = ['income statement', 'revenue', 'sales', 'gross margin', 'operating expenses', 'cost of goods sold',
                              'research and development', 'net income', 'income', 'depreciation', 'tax', 'taxes']
    
    bs_score = 0  # Tracks Balance Sheet keyword matches
    is_score = 0  # Tracks Income Statement keyword matches

    for line in cleaned:
        text = line.get_text()

        # Score balance sheet terms
        for b_term in BALANCE_SHEET_TERMS:
            if b_term in text.lower():
                bs_score += 1
        
        # Score income statement terms
        for i_term in INCOME_STATEMENT_TERMS:
            if i_term in text.lower():
                is_score += 1
    
    # Decision logic
    if bs_score > is_score and bs_score >= score_thresh:
        print(f'BS identified. BS score: {bs_score} IS score: {is_score}')
        return 'BALANCE_SHEET'
    elif is_score > bs_score and is_score >= score_thresh:
        print(f'IS identified. IS score: {is_score} BS score: {bs_score}')
        return 'INCOME_STATEMENT'
    else:
        raise ValueError('Could not recognize document')


def merge_lines(lines, debug=False):
    """
    Merges fragmented line items in financial statements based on formatting cues.
    A merge is performed only if:
    - The current line's label does not start with a capitalized word (likely a continuation).
    - The current line is indented farther than the previous (child of previous).
    - One of the two lines has values, but not both (to avoid overwriting).

    Args:
        lines (list of RawData): Parsed lines from the OCR output.
        debug (bool): If True, prints detailed merge logic steps.

    Returns:
        list of RawData: A cleaned list of lines, with appropriate lines merged together.
    """

    output = []

    for line in lines:
        label, _, vals, d_sign, indent, _ = line.get_all()

        # Skip merging if there's no label at all
        if not label:
            output.append(line)
            continue

        first_char = label[0]

        # Checks if continuation line - continuation lines should not start with a capitalized letter
        if not (first_char.isalpha() and first_char.isupper()):
            if not output:
                output.append(line)
                continue

            prev = output[-1]
            prev_label, _, prev_vals, _, prev_indent, _ = prev.get_all()
            
            # Only merge if current line is indented more than the previous
            if prev_indent < indent:
                if (vals and prev_vals) or (not vals and not prev_vals):  # Prevent merge if both lines have values or neither do
                    if debug:
                        print(f'Tried to merge but ran into Error. At least one line must have vals but not both. Lines: {label} | {prev_label}')
                    output.append(line)
                    continue
                full_label = prev_label.strip() + ' ' + label.strip()
                if debug:
                    print(f'MERGIING LINE. LINE NOW: {full_label}')

                # if continuation line has data, transfer to line above
                if vals:
                    prev.vals = vals
                if d_sign:
                    prev.add_dollar_sign()
                prev.add_text(full_label)
            else:
                # failed indentation check
                if debug:
                    print(f'Warning. Line: {label} is not capitalized but could not merge')
                output.append(line)
        else:
            # Handles normal lines
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
    Builds a FinancialStatement object from parsed OCR line data.
    - Determines the financial statement type (Balance Sheet or Income Statement).
    - Extracts year labels from the header (e.g., "2022 2021").
    - Iterates over each parsed line and builds a LineItem object with associated values.
    - Matches each value to the closest column coordinate (representing each year).
    - Adds headings or unquantified labels directly to the financial statement.

    Args:
        col_coords (list of int): X-coordinates representing columns of values (one per year).
        lines (list of RawData): Lines parsed from OCR with detected labels/values/dollar signs.
        debug (bool): If True, prints detailed diagnostic information.
        val_x_thresh (int): Maximum allowed distance from a column coordinate for value matching.

    Returns:
        FinancialStatement: An object containing structured data parsed from the OCR output.
    """
    extract_date = re.compile(r'^(?:\D*?(?:19|20)\d{2}){2,}\D*$')
    year_pattern = re.compile(r'(?:19|20)\d{2}')

    got_years = False
    end_collection = False
    new_fs = FinancialStatement()

    # Infer financial statement type and stopping point
    fs_type = what_fs(lines)
    end = get_end(fs_type)

    # Default years in case years not found
    years = ['Year ' + str(num + 1) for num in range(len(col_coords))]

    for line in lines:
        label = line.get_text()
        label = label.strip('.').strip('_')  # for formats where labels and vals separated with periods
        dollar_sign = line.get_dollar_sign()

        # If years found, extract years and delete useless lines before years
        if not got_years and extract_date.search(label):
            matches = year_pattern.findall(label)
            years = [int(y) for y in matches]
            got_years = True
            new_fs = FinancialStatement()
            if debug:
                print('CAPTURED YEARS:', years)
            continue

        # End detected and line does not match end regex. Exclude useless lines
        if end_collection and not end.match(label):
            if debug:
                print(f'Final line detected. Excluded: {label}')
            break

        new_line_item = LineItem()

        if dollar_sign:
            new_line_item.add_dollar_sign()

        # Checks if line item with vals
        vals = line.get_vals()
        if not line.get_vals() == []:

            # Sets end_collection flag to true. Sometimes multiple lines match end collection and we want the last one which is why the loop does not stop here. 
            if end.match(label) and dollar_sign:
                if debug:
                    print(f'Final line detected and included: {label}')
                end_collection = True

            if debug:
                print(f'LINE ITEM DETECTED: {label}')

            assigned_vals = [None] * len(col_coords)

            for val, val_coord in vals:
                # Find the closest column index based on X distance
                distances = [abs(val_coord - col_x) for col_x in col_coords]
                closest_idx = np.argmin(distances)

                # Confirms detected val is a val and not cut off from label. Happens when label ends with number
                if distances[closest_idx] <= val_x_thresh:
                    if detect_vals.match(val):
                        cleaned = val.replace(',', '').replace('_', '').replace('(', '-').replace(')', '') # Remove punctuation. Mark as negative with - if val in ()s
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
                    # If val was probably misaligned and should be part of the label
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

            # Fill in values for each expected year. Pad with 0s if necessary
            for idx, year in enumerate(years):
                value = assigned_vals[idx] if assigned_vals[idx] is not None else 0
                new_line_item.add_data(year, value)

        else:
            if end_collection:  # It may seem like this check isnt necessary but its important esp in IS with "net income per share" lines since they trigger the regex but dont have vals and shouldnt be included since final net income already captured
                if debug:
                   print(f'END REACHED. EXCLUDED: {label}')
                break

            if debug:  # Checks for headings without vals
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
        self.y_vals = (None, None)
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

    def add_y_vals(self, y1, y2):
        self.y_vals = (y1, y2)

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
    
    def get_y_vals(self):
        return self.y_vals

    def get_all(self):
        return self.text, self.text_x_coords, self.vals, self.dollar_sign, self.indent, self.y_vals

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