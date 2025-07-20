from enum import Enum, auto

class RawLine:
    def __init__(self):
        self.label = None #text of the line 
        self.text_x_coords = (None, None) # min and max x of the bounding box 
        self.dollar_sign = False # aesthetically does it need a dollar sign 
        self.indent = 0 # how indented is it (step size of 1)
        self.y_vals = (None, None) # min and max y of the bounding box
        self.val_y_vals = [] # List of max y for value bboxes on a given line
        self.is_total = False  # bool of whether line is total as indicated by summing line above it
        self.vals = []  # List of (financial value, x_coord) pairs

    def add_label(self, label):
        assert isinstance(label, str), "Text must be a string"
        self.label = label

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
    
    def add_val_y_val(self, val):
        self.val_y_vals.append(val)
    
    def set_total(self):
        self.is_total = True

    def get_text(self):
        return self.label

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
    
    def get_val_y_vals(self):
        return self.val_y_vals

    def get_all(self):
        return self.label, self.text_x_coords, self.vals, self.dollar_sign, self.indent, self.y_vals

    def __str__(self):
        return f"TEXT: {self.label} | COORDS: {self.text_x_coords} | VALS: {self.vals}"


class LineItem:
    def __init__(self):
        self.label = None # Text of the line
        self.dollar_sign = False  # aesthetically does it need a dollar sign 
        self.indent = 0 # how indented is it (step size of 1)
        self.is_total = False
        self.summing_type = 0  
        self.summing_range = None
        self.data = {}

    def add_label(self, label):
        assert isinstance(label, str)
        self.label = label

    def add_data(self, year, value=None):
        assert isinstance(value, (int, float))
        self.data[year] = value

    def add_dollar_sign(self):
        self.dollar_sign = True
    
    def add_indent(self, indent):
        self.indent = indent

    def add_summing_type(self, val):
        self.summing_type = val

    def add_summing_range(self, lst):
        self.summing_range = lst
    
    def set_total(self):
        self.is_total = True
        
    def get_label(self):
        return self.label

    def get_dollar_sign(self):
        return self.dollar_sign

    def get_indent(self):
        return self.indent

    def get_data(self):
        return self.data

    def get_dollar_sign(self):
        return self.dollar_sign

    def get_summing_type(self):
        return self.summing_type

    def get_summing_range(self):
        return self.summing_range

    def get_all(self):
        return self.label, self.data, self.dollar_sign, self.indent, self.summing_type, self.summing_range

    def __str__(self):
        data_pairs = [(year, self.data[year]) for year in sorted(self.data.keys())]
        return f'Line item: {self.label} | Values: {data_pairs}'

class FinancialStatement:
    def __init__(self):
        self.lines = []
        self.fs_type = None
    
    def add_type(self, fs_type):
        self.fs_type = fs_type

    def add_line_item(self, data):
        assert isinstance(data, (LineItem))
        self.lines.append(data)
    
    def add_years(self, years):
        self.years = years

    def get_lines(self):
        return self.lines
    
    def get_type(self):
        return self.fs_type
    
    def get_years(self):
        return self.years

    def __str__(self):
        return "\n".join(str(line) for line in self.lines)
    
class State(Enum):
    INIT = auto()
    LOADED_DATA = auto()
    PREPROCESSED = auto()
    MERGED = auto()
    COMPLETED = auto()
    ERROR = auto()

class Format(Enum):
    CSV = auto()
    XLSX = auto()
