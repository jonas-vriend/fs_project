import easyocr
import cv2
import re
import pickle
import os
import csv
import numpy as np
from pdf2image import convert_from_path
from PIL import Image, ImageOps 

pdf_path = "bs_23.pdf"
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
    # Convert to OpenCV format
    img = np.array(pil_image)

    # Binary image
    _, binary = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)

    # Create horizontal kernel and detect horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    detected_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    color_img[detected_lines > 0] = [0, 0, 255]  # red lines

    Image.fromarray(color_img).save(os.path.join(os.getcwd(), "debug_detected_lines.png"))
    # Subtract lines from the original
    cleaned = cv2.bitwise_not(binary)
    cleaned = cv2.bitwise_or(cleaned, detected_lines)

    # Convert back to PIL
    cleaned_rgb = cv2.cvtColor(cv2.bitwise_not(cleaned), cv2.COLOR_GRAY2RGB)
    return Image.fromarray(cleaned_rgb)

if os.path.exists(cache_file):
    os.remove(cache_file) 
    #with open(cache_file, "rb") as f:
     #   results = pickle.load(f)
    #print(f"Loaded OCR results from cache file: {cache_file}")
    reader = easyocr.Reader(['en'])
    processed = remove_horizontal_lines(larger)
    processed.save("cleaned_no_lines.png")
    results = reader.readtext("cleaned_no_lines.png", width_ths=70)
    with open(cache_file, "wb") as f:
        pickle.dump(results, f)
else:
    reader = easyocr.Reader(['en'])
    processed = remove_horizontal_lines(larger)
    processed.save("cleaned_no_lines.png")
    results = reader.readtext("cleaned_no_lines.png", width_ths=70)
    with open(cache_file, "wb") as f:
        pickle.dump(results, f)

raw_text = [text for _, text, _ in results]

def build_bs(data, debug=False):
    cleaned = []
    unwanted_word = re.compile(r'^[a-z]{1,9}$') # checks for standalone short lowercase words that are likely noise
    suspicious_digits = re.compile(r'\b\d{1,3},\d{1,2}\b') # checks for standalone digits that are likely noise
    end = re.compile(r'(?i)total.*liabilit(?:y|ies).*equity.*\d+')
    line_item = re.compile(r"""(?ix)
        ^([A-Za-z0-9\s\-',();:&$\.]+?)          # (1) label
        \s*                                     
        (                                       # (2) one or more values
            (?:\s*\$?\s*\(?\d{1,3}(?:,\d{3})*\)?)+
        )$
    """)

    continuation = re.compile(r"""(?x)
        ^[^A-Z]                                      # must NOT start with uppercase
        [A-Za-z0-9\s\-',();:&$\.]*?                  # label text
        (?:\s*\$?\s*\(?\d{1,3}(?:,\d{3})*\)?)+$       # ends in at least one numeric value
    """)


    extract_date = re.compile(r'^(?:\D*?(?:19|20)\d{2}){2,}\D*$')
    year_pattern = re.compile(r'(?:19|20)\d{2}')

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
            if not line_item.match(prev) and line_item.match(stripped) and len(prev) > 40:
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
    years = set()
    for line in cleaned:
        if not got_years:
            found_years = extract_date.search(line)
            if found_years:
                matches = year_pattern.findall(line)
                years.update(int(y) for y in matches)
                years = sorted(years, reverse=True)
                got_years = True
                continue
            else:
                if debug:
                    print('skipped line before date captured:', line)
                continue

        match = line_item.match(line)
        if match:
            label = match.group(1).strip()
            vals = re.findall(r'\(?\d{1,3}(?:,\d{3})*\)?', match.group(2))
            cleaned_vals = [float(val.replace(',', '').replace('(', '-').replace(')', '').replace('$', '')) for val in vals]
            new_line_item = LineItem(label)
            for year, val in zip(years, cleaned_vals):
                new_line_item.add_data(year, val)
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