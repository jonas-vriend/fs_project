import easyocr
import cv2
import re
import pickle
import os
import csv
import numpy as np
from pdf2image import convert_from_path
from PIL import Image, ImageOps 

pdf_path = "Amazon_is_22.pdf"
images = convert_from_path(pdf_path, dpi=300)
pdf_basename = os.path.splitext(os.path.basename(pdf_path))[0]
image_path = f"{pdf_basename}_page1.png"
image = images[0]
image.save(image_path)

higher_clarity = Image.open(image_path).convert("L") 
higher_clarity = ImageOps.autocontrast(higher_clarity)

scale_factor = 2  
larger = higher_clarity.resize(
    (higher_clarity.width * scale_factor, higher_clarity.height * scale_factor)
)
larger.save("high_contrast_page.png")

image_filename = os.path.splitext(os.path.basename(image_path))[0]
cache_file = f"ocr_cache_{image_filename}.pkl"

def remove_horizontal_lines(pil_image):
    # Convert PIL to OpenCV grayscale
    img = np.array(pil_image)

    # Binarize image (invert for easier line detection)
    _, binary = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)

    # Detect horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    detected_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)

    # Visualize horizontal lines on original for debugging
    color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    color_img[detected_lines > 0] = [0, 255, 0]  # green for detected lines
    debug_line_path = "debug_detected_lines.png"

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

    else:
        reader = easyocr.Reader(['en'])
        processed, debug_path = remove_horizontal_lines(larger)
        processed.save("cleaned_no_lines.png")
        print(f"Debug image saved to: {debug_path}")
        results = reader.readtext("cleaned_no_lines.png", width_ths=70)
        with open(cache_file, "wb") as f:
            pickle.dump(results, f)
    
    return results

results = get_data()
raw_text = [text for _, text, _ in results]

def what_fs(cleaned):
    BALANCE_SHEET_TERMS = ['balance sheet', 'asset', 'assets', 'liability',
                            'liabilities', 'inventory', 'inventories', 'property', 'plant', 'equipment',
                            'accounts payable', 'deferred revenue', "shareholder's equity", 'common stock', 'accounts receivable',
                            'additional paid-in capital', 'accumulated other comprehensive income', 'cash and cash equivalents', 'retained earnings']
    
    INCOME_STATEMENT_TERMS = ['income statement', 'revenue', 'sales', 'gross margin', 'operating expenses', 'cost of goods sold',
                              'research and development', 'net income', 'income', 'depreciation', 'tax', 'taxes']
    
    bs_score = 0
    is_score = 0
    for line in cleaned:
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
        return re.compile(r'(?i)net.*income.*(?:\(loss\))')

# Fixes problem in amazon bs 24. This may not be a fool proof way to handle regex edge cases. Consider revising in the future. 
def parentheses_check(vals):
    for val in vals:
        if ('(' in val and ')' not in val ) or (')' in val and '(' not in val):
            return True
    return False

def build_fs(data, debug=False):
    cleaned = []
    malformed = []
    unwanted_word = re.compile(r'^[a-z]{1,9}$') # checks for standalone short lowercase words that are likely noise
    suspicious_digits = re.compile(r'\b\d{1,3},\d{1,2}\b') # checks for standalone digits that are likely noise
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

    for line in data:
        stripped = line.strip().replace('_', '')
        if debug:
            print(f"STRIPPED LINE: '{stripped}'")
        if unwanted_word.match(stripped):
            if debug:
                print(f'caught unwanted word: {stripped}')
            continue
        elif suspicious_digits.match(stripped):
            if debug:
                print(f'caught suspicious digits: {stripped}')
            continue
        elif continuation.match(stripped) and cleaned is not []:
            prev = cleaned[-1]
            if not line_item.match(prev) and line_item.match(stripped):# and len(prev) > 40:
                if debug:
                    print(f"MERGING '{prev}' + '{stripped}'")
                cleaned[-1] = prev + ' ' + stripped
                continue
            else:
                cleaned.append(stripped)
        else:
            cleaned.append(stripped)

    fs_type = what_fs(cleaned)
    new_fs = FinancialStatement()

    got_years = False
    end = get_end(fs_type)
    for line in cleaned:
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
            if parentheses_check(vals):
                if debug:
                    print(f'Parentheses check failed. Treaing line as header: {line}')
                    new_line_item = LineItem(line)
                    new_fs.add_line_item(new_line_item)
                    continue
            cleaned_vals = [float(val.replace(',', '').replace('(', '-').replace(')', '').replace('$', '')) for val in vals]
            if len(cleaned_vals) < num_years:
                if debug:
                    print(f'Added {(num_years - len(cleaned_vals))} 0(s) to line: {label}')
                cleaned_vals = cleaned_vals + [0.0] * (num_years - len(cleaned_vals))
                malformed.append(label)
            new_line_item = LineItem(label)
            for year, val in zip(years, cleaned_vals):
                new_line_item.add_data(year, val)
            new_fs.add_line_item(new_line_item)
        else:
            if debug:
                print('Did not recognize as Line Item:', line)
            new_line_item = LineItem(line)
            new_fs.add_line_item(new_line_item)
        
        if end.match(line):
            if debug:
                print('Ending at line: ', line)
            break

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


def debug_output(data, verbose=False):
    img = cv2.imread("cleaned_no_lines.png")

    for bbox, text, conf in data:
        if verbose:
            print(f"{text} (confidence: {conf:.2f})")
        bbox = [tuple(map(int, point)) for point in bbox]
        cv2.rectangle(img, bbox[0], bbox[2], (0, 255, 0), 2)
        cv2.putText(img, text, bbox[0], cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    cv2.imwrite("debug_overlay.png", img)


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

result = build_fs(raw_text, True)
export_fs(result)
debug_output(results)
