import easyocr
import cv2
import re
import pickle
import os
import csv
from pdf2image import convert_from_path
from PIL import Image, ImageOps 

pdf_path = "bs_15.pdf"
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


def build_bs(data, debug=False):
    cleaned = []
    unwanted_word = re.compile(r'^[a-z]{1,9}$') # checks for standalone short lowercase words that are likely noise
    suspicious_digits = re.compile(r'\b\d{1,3},\d{1,2}\b') # checks for standalone digits that are likely noise
    end = re.compile(r'(?i)total.*liabilit(?:y|ies).*equity.*\d+')
    line_item = re.compile(r"""(?ix)
        ^([A-Za-z0-9\s\-',();:&$\.]+?)        # label
        \s*
        (?:\$?\s*)?
        (\(?\d{1,3}(?:,\d{3})*\)?)          # value 1
        \s*
        (?:\$?\s*)?
        (\(?\d{1,3}(?:,\d{3})*\)?)$         # value 2
    """)
    continuation = re.compile(r"""(?ix)
        ^[a-z]                                   # must start lowercase
        [A-Za-z0-9\s\-',();:&$\.]*?              # label text
        \s*                                      # space
        (?:\$?\s*)?                              # optional dollar sign and space
        (\(?\d{1,3}(?:,\d{3})*\)?)               # first number
        \s*                                      # space
        (?:\$?\s*)?                              # optional dollar sign and space
        (\(?\d{1,3}(?:,\d{3})*\)?)$              # second number
    """)
    extract_date = re.compile(r'(\d{4}).*(\d{4})')
    for line in data:
        stripped = line.strip().replace('_', '')
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
        elif continuation.match(stripped) and cleaned is not []:
            prev = cleaned[-1]
            if not line_item.match(prev):
                if debug:
                    print(f"MERGING '{prev}' + '{stripped}'")
                cleaned[-1] = prev + ' ' + stripped
                continue
            else:
                cleaned.append(stripped)
        else:
            cleaned.append(stripped)

    new_bs = BalanceSheet()

    got_years = False
    y1, y2 = None, None
    for line in cleaned:
        if not got_years:
            found_years = extract_date.search(line)
            if found_years:
                y1 = int(found_years.group(1))
                y2 = int(found_years.group(2))
                if debug:
                    print('years found:', y1, y2)
                got_years = True
                continue
            else:
                if debug:
                    print('skipped line before date captured:', cleaned)
                continue

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

def export_balance_sheet(bs, filename="balance_sheet.csv"):

    all_years = sorted(
        {year for item in bs.lines for year in item.data.keys()},
        reverse=True
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
    img = cv2.imread("high_contrast_page.png")

    for bbox, text, conf in data:
        if verbose:
            print(f"{text} (confidence: {conf:.2f})")
        bbox = [tuple(map(int, point)) for point in bbox]
        cv2.rectangle(img, bbox[0], bbox[2], (0, 255, 0), 2)
        cv2.putText(img, text, bbox[0], cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    cv2.imwrite("debug_overlay1.png", img)


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

result = build_bs(raw_text, True)
export_balance_sheet(result)
debug_output(results)