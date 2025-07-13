from enum import Enum, auto

class RawLine:
    def __init__(self):
        self.text = None
        self.text_x_coords = (None, None)
        self.dollar_sign = False
        self.indent = 0
        self.y_vals = (None, None)
        self.vals = []  # List of (financial value, x_coord) pairs

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
    
class State(Enum):
    INIT = auto()
    LOADED_DATA = auto()
    PREPROCESSED = auto()
    MERGED = auto()
    COMPLETED = auto()
    ERROR = auto()
