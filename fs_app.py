import easyocr
import cv2
import re
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

reader = easyocr.Reader(['en'])
results = reader.readtext(
    "high_contrast_page.png",
    width_ths = 70
    )

raw_text = [text for _, text, _ in results]
print("===RAW TEXT===\n", raw_text)
print('\n')
def clean_data(data, debug=False):
    cleaned = []
    unwanted_word = re.compile(r'^[^A-Z][a-z]{1,9}$')
    suspicious_digits = re.compile(r'\b\d{1,3},\d{1,2}\b')
    end = re.compile(r'(?i)total.*liabilit(?:y|ies).*equity.*\d+')
    for line in data:
        stripped = line.strip()
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


    

def debug_output(data, verbose=False):
    img = cv2.imread("high_contrast_page.png")

    for bbox, text, conf in data:
        if verbose:
            print(f"{text} (confidence: {conf:.2f})")
        bbox = [tuple(map(int, point)) for point in bbox]
        cv2.rectangle(img, bbox[0], bbox[2], (0, 255, 0), 2)
        cv2.putText(img, text, bbox[0], cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    cv2.imwrite("debug_overlay1.png", img)


debug_output(results)
cleaned = clean_data(raw_text, True)
print(cleaned)
if '13,5' in raw_text:
    print('found in raw')
if '13,5' in cleaned:
    print('found in cleaned')

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
    
class IncomeStatement:
    def __init__(self):
        self.lines = []

    def add_line_items(self, data):
        assert isinstance(data, (LineItem))
        self.lines.append(data)

test = LineItem('D&A')
test.add_data(2025, 1000)
test.add_data(2024, 1000)
test.add_data(2023, 1000)
# print(test)