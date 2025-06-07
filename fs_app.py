from PIL import Image
from pdf2image import convert_from_path
import pytesseract

pages = convert_from_path("apple_bs.pdf", dpi=300)


for i, page in enumerate(pages):
    text = pytesseract.image_to_string(page)
    print(f"\n=== Page {i+1} ===\n{text}")


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
    def __init__(self, data):
        self.lines = []

    def add_line_items(self, data):
        assert isinstance(data, (LineItem))
        self.lines.append(data)

test = LineItem('D&A')
test.add_data(2025, 1000)
test.add_data(2024, 1000)
test.add_data(2023, 1000)
# print(test)