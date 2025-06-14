import easyocr
import cv2
import re
import pickle
import os
from pdf2image import convert_from_path
from PIL import Image, ImageOps 

images = convert_from_path("apple_bs.pdf", dpi=300)
image = images[0]
image_path = "temp_page.png"
image.save(image_path)

higher_clarity = Image.open(image_path).convert("L") 
higher_clarity = ImageOps.autocontrast(higher_clarity)

scale_factor = 2  
larger = higher_clarity.resize(
    (higher_clarity.width * scale_factor, higher_clarity.height * scale_factor)
)
larger.save("high_contrast_page.png")

image_filename = os.path.basename(image_path).replace(".png", "")
cache_file = f"ocr_cache_{image_filename}.pkl"

if os.path.exists(cache_file):
    with open(cache_file, "rb") as f:
        results = pickle.load(f)
    print(f"Loaded OCR results from cache file: {cache_file}")
else:
    reader = easyocr.Reader(['en'])
    results = reader.readtext("high_contrast_page.png", width_ths=70)
    with open(cache_file, "wb") as f:
        pickle.dump(results, f)

raw_text = [text for _, text, _ in results]
#print("===RAW TEXT===\n", raw_text)
#print('\n')

def clean_data(data, debug=False):
    cleaned = []
    unwanted_word = re.compile(r'^[^A-Z][a-z]{1,9}$')
    suspicious_digits = re.compile(r'\b\d{1,3},\d{1,2}\b')
    end = re.compile(r'(?i)total.*liabilit(?:y|ies).*equity.*\d+')
    for line in data:
        stripped = line.strip()
        if debug:
            print(f"LINE: '{stripped}'")
        if unwanted_word.match(stripped):
            if debug:
                print(f'caught unwanted word: {stripped}')
            continue
        elif suspicious_digits.match(stripped):
            if debug:
                print(f'caught suspicious digits: {stripped}')
            continue
        elif end.match(stripped):
            cleaned.append(stripped)
            break
        else:
            cleaned.append(stripped)
    return cleaned


def get_date(clean_data):
    extract_date = re.compile(r'(\d{4}).*(\d{4})')
    for line in clean_data:
        found_years = extract_date.search(line)
        if found_years:
            y1 = found_years.group(1)
            y2 = found_years.group(2)
            return (int(y1), int(y2))
    return (None, None)


def build_bs(clean_data):
    new_bs = BalanceSheet()
    line_item = re.compile(r"""(?ix)
    ^([A-Za-z0-9\s\-',();:&$.]+?)            # (1) Label 
    \s*                                      # Space between label and first number
    (\(?\d{1,3}(?:,\d{3})*\)?)               # (2) First number (with optional parentheses)
    \s*                                      # Optional spacing
    (\(?\d{1,3}(?:,\d{3})*\)?)$              # (3) Second number (with optional parentheses)
    """)

    y1, y2 = get_date(clean_data)

    for line in clean_data:
        match = line_item.match(line)
        if match:
            label = match.group(1).strip()
            try:
                y1_val = float(match.group(2).replace(',', '').replace('(', '-').replace(')', '').replace('$', ''))
                y2_val = float(match.group(3).replace(',', '').replace('(', '-').replace(')', '').replace('$', ''))
            except ValueError:
                print(f"Warning: Could not parse line: {line}")
                continue
            new_line_item = LineItem(label)
            new_line_item.add_data(y1, y1_val)
            new_line_item.add_data(y2, y2_val)
            new_bs.add_line_item(new_line_item)
        else:
            new_line_item = LineItem(line)
            new_bs.add_line_item(new_line_item)

    return new_bs


def debug_output(data, verbose=False):
    img = cv2.imread("high_contrast_page.png")

    for bbox, text, conf in data:
        if verbose:
            print(f"{text} (confidence: {conf:.2f})")
        bbox = [tuple(map(int, point)) for point in bbox]
        cv2.rectangle(img, bbox[0], bbox[2], (0, 255, 0), 2)
        cv2.putText(img, text, bbox[0], cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    cv2.imwrite("debug_overlay1.png", img)

cleaned = clean_data(raw_text)

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
        data_pairs = [(year, self.data[year]) for year in sorted(self.data.keys(), reverse=True)]
        return f'Line item: {self.name} | Values: {data_pairs}'
    
class BalanceSheet:
    def __init__(self):
        self.lines = []

    def add_line_item(self, data):
        assert isinstance(data, (LineItem))
        self.lines.append(data)

    def __str__(self):
        return "\n".join(str(line) for line in self.lines)

result = build_bs(cleaned)
print(result)