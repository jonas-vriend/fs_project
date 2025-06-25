import easyocr
import cv2
import re
import pickle
import os
import csv
import numpy as np
from pdf2image import convert_from_path
from PIL import Image, ImageOps 

# Get the input PDF path
pdf_path = os.path.join("Financials", "IS", "Walmart_is_24.pdf")

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
    _, binary = cv2.threshold(img, 160, 255, cv2.THRESH_BINARY_INV) # Very important: first number is threshold for whther something is treated as white or black which can delete entire sections if not careful

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
        results = reader.readtext(np.array(processed), width_ths=70)
        with open(cache_file, "wb") as f:
            pickle.dump(results, f)
    
    return results, processed

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
        return re.compile(r'(?i).*net income.*(?:\(loss\))?.*') # need to make this more robust. 

def build_fs(data, debug=False):
    cleaned = []
    malformed = []
    rborders = []
    line_item = re.compile(r"""(?ix)
        ^([A-Za-z0-9\s\-',();:&$/\.]+?)          # (1) label
        \s*                                     
        (                                        # (2) one or more values
            (?:\s*\$?\s*\(?\d[\d,]*\)?)+         # only digits+commas inside optional ()
        )\s*$
    """)

    continuation = re.compile(r"""(?x)
        ^[^A-Z]                                      # must NOT start with uppercase
        [A-Za-z0-9\s\-',();:&$\.]*?                  # label text
        (?:\s*\$?\s*\(?[\d,]+\)?)+$                  # ends in at least one numeric value (with or without commas)
    """)

    extract_date = re.compile(r'^(?:\D*?(?:19|20)\d{2}){2,}\D*$')
    year_pattern = re.compile(r'(?:19|20)\d{2}')

    for bbox, line, _ in data:
        _, tr, _, _ = bbox
        x, _ = tr
        stripped = line.strip() # removed .replace('_', '')
        if debug:
            print(f"STRIPPED LINE: '{stripped}'")

        if line_item.match(stripped):
            rborders.append(x)

        if continuation.match(stripped) and cleaned is not []:
            prev, _ = cleaned[-1]

            if not line_item.match(prev) and line_item.match(stripped):# and len(prev) > 40:
                if debug:
                    print(f"MERGING '{prev}' + '{stripped}'")
                combined = prev + ' ' + stripped
                cleaned[-1] = (combined, x)
                continue
            else:
                cleaned.append((stripped, x))
        else:
            cleaned.append((stripped, x))

    line_item_threshold = np.median(rborders)
    if debug:
        print(f'Line item threshold found: {line_item_threshold}')
    fs_type = what_fs(cleaned)
    new_fs = FinancialStatement()

    got_years = False
    end = get_end(fs_type)
    for line, x in cleaned:
        if not got_years:
            found_years = extract_date.search(line)
            if found_years:
                matches = year_pattern.findall(line)
                years = [int(y) for y in matches]
                num_years = len(years)
                got_years = True
                if debug:
                    print('captured years', years)
                continue
            else:
                if debug:
                    print('skipped line before date captured:', line)
                continue
        match = line_item.match(line)
        if match:
            if debug:
                print('got line item match:', line)
            label = match.group(1).strip()
            vals = re.findall(r'\(?[\d,]+\)?', match.group(2))
            TOLERANCE = 50
            if x > line_item_threshold + TOLERANCE or x < line_item_threshold - TOLERANCE:
                if debug:
                    print(f'x coord check failed. x coord: {x} Treating line as header: {line}' )
                    new_line_item = LineItem(line)
                    new_fs.add_line_item(new_line_item)
                    continue
            cleaned_vals = [float(val.replace(',', '').replace('(', '-').replace(')', '').replace('$', '')) for val in vals]
            if len(cleaned_vals) < num_years:
                if debug:
                    print(f'Added {(num_years - len(cleaned_vals))} 0(s) to line: {label}')
                cleaned_vals = cleaned_vals + [0.0] * (num_years - len(cleaned_vals))
                malformed.append(label)
            if len(cleaned_vals) > num_years:
                if debug:
                    print(f'More vals than years detected in line: {line}')
                excess = len(cleaned_vals) - num_years
                discarded_text = cleaned_vals[:excess]
                discarded_text_to_str = [str(int(val)) for val in discarded_text]
                label += " " + " ".join(discarded_text_to_str)
                cleaned_vals = cleaned_vals[-num_years:]
            new_line_item = LineItem(label)
            for year, val in zip(years, cleaned_vals):
                new_line_item.add_data(year, val)
            new_fs.add_line_item(new_line_item)
            if end.match(line):
                if debug:
                    print('Ending at line: ', line)
                break
        else:
            if debug:
                print('Did not recognize as Line Item:', line)
            new_line_item = LineItem(line)
            new_fs.add_line_item(new_line_item)

    return new_fs

def export_fs(bs, filename=f"financial_statement.csv"):

    all_years = sorted(
        {year for item in bs.lines for year in item.data.keys()}
    )

    with open(filename, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Label"] + all_years)

        for item in bs.lines:
            row = [item.name]
            for year in all_years:
                value = item.data.get(year, "")
                row.append(value)
            writer.writerow(row)

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
result = build_fs(ocr_output, True)
export_fs(result)
debug_output(ocr_output, processed_img)

