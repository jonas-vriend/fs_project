import easyocr
import cv2
import re
import pickle
import os
import numpy as np
from collections import Counter
from ..models import RawLine, FinancialStatement, LineItem, State, Format
from ..export import export_fs_as_csv, export_fs_as_xlsx
from .base import BaseProcessor
from ..utils import preprocess_img, get_coords, get_all_subsets, find_solution


########################### GLOBAL VARIABLES ###################################

detect_vals = re.compile(r'^\(?-?[$S]?\s?\d{1,3}(?:,\d{3})*(?:\.\d+)?\)?$')  # Regex to detect financial values on right side of financial statement

################################################################################


class OcrProcessor(BaseProcessor): 

    def __init__(self, pdf_path, debug=False, use_cache=False, export_filename="financial_statement"):
        super().__init__(pdf_path, debug, use_cache, export_filename)
        
        self.state = State.INIT  # Runtime state
        self.img = None
        self.ocr_output = None
        self.underscore_coords = None
        self.summing_line_coords = None
        self.col_coords = None
        self.raw_lines = None
        self.merged_lines = None
        self.years = []
        self.fs = None  # Final FinancialStatement object
    
    def process(self):
        """
        Orchestrates the pipeline:
        - Load/caches OCR
        - Extracts structure and values
        - Builds financial statement object
        - Exports as CSV
        """
        self.get_data() #state becomes LOADED_DATA
        assert self.img is not None and self.img.size != 0, "Image not loaded correctly."
        assert self.ocr_output is not None and len(self.ocr_output) > 0, "OCR output is empty or missing."
        assert self.underscore_coords is not None, "Underscore coordinates not set."
        print(self.state)

        self.get_x_bounds()
        assert self.col_coords is not None and len(self.col_coords) > 0, "No columns detected"

        self.preprocess_text() #state becomes PREPROCESSED
        assert self.raw_lines is not None and len(self.raw_lines) > 0, "No text lines detected"
        print(self.state)

        if self.underscore_coords:
            self.add_underscore_zeros()
        if self.summing_line_coords:
            self.assign_summing_lines()

        self.debug_output(val_x_thresh=75)
        self.add_indentation()

        self.merge_lines()  #state becomes MERGED
        print(self.state)
        assert self.merged_lines is not None and len(self.merged_lines) > 0, "Merging lines resulted in empty output."

        self.build_fs() #state becomes COMPLETED
        print(self.state)
        assert self.fs is not None and len(self.fs.lines) > 0, "Financial statement build failed: no lines added"

        self.get_summing_ranges()


    def get_data(self):
        """
        Runs the OCR pipeline on the current PDF.

        If a cached OCR result exists for the PDF, it is loaded from disk to save time.
        Otherwise, the image is processed, OCR is run using EasyOCR, and the results are cached.

        Returns:
            - results: the OCR output (list of [bbox, text, confidence] from EasyOCR)
            - processed: the final image after preprocessing (PIL.Image)
            - underscore_coords: list of bounding boxes for detected underscore "0" lines
        """
        assert self.state == State.INIT

        # Preprocess the PDF and detect underscore lines and debug overlay path
        img, underscore_coords, summing_line_coords = preprocess_img(self.pdf_path, self.debug)
        self.img = img
        self.underscore_coords = underscore_coords
        self.summing_line_coords = summing_line_coords

        # Create filename for cache 
        image_filename = os.path.splitext(os.path.basename(self.pdf_path))[0]
        cache_file = os.path.join("Cache", f"ocr_cache_{image_filename}.pkl")

        # If cache exists and allowed, load results from disk
        if self.use_cache and os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                ocr_output = pickle.load(f)
            print(f"Loaded OCR results from cache file: {cache_file}")

        else:
        
            # Otherwise, run OCR from scratch
            reader = easyocr.Reader(['en'])
            
            ocr_output = reader.readtext(
                np.array(img), 
                width_ths=0.5,  # Controls how aggressively EasyOCR merges text into the same box (0.5 results in many boxes) (TODO)
                text_threshold=0.55   # Allows text with   score to be captured. Lowering this improves recongition of single character values but also makes OCR hallucinate more (TODO)
            )

            # Save OCR results to cache for future runs
            with open(cache_file, "wb") as f:
                pickle.dump(ocr_output, f)

        self.ocr_output = ocr_output
        self.state = State.LOADED_DATA


    def get_x_bounds(self, y_thresh=50):
        """
        Takes list of coordinates of detected values in each row and takes median value
        of the right bbox border x values for each column as a guide for where the values should be
        """
        assert self.state == State.LOADED_DATA

        num_vals = []  #Stores how many values per line (to determine how many columns to expect)
        all_val_coords = []  # list of lists storing x2 coords of each value on each line (to determine column divider)
        line_val_coords = []  # Holds x2 coords of values on current line
        
        # Initialize y
        for i, (bbox, text, _) in enumerate(self.ocr_output):
            if detect_vals.match(text):
                _, x2, y1, _ = get_coords(bbox)
                line_val_coords.append(x2)
                current_y = y1
                start = i + 1
                break

        # Find x2 coords of detected vals
        for bbox, text, _ in self.ocr_output[start:]:
            if detect_vals.match(text):
                _, x2, y1, _ = get_coords(bbox)
                if abs(y1 - current_y) < y_thresh:
                    line_val_coords.append(x2)
                else:
                    # New line detected. Append info to relevant lists; reset everything for new line
                    num_vals.append(len(line_val_coords))
                    all_val_coords.append(line_val_coords)
                    current_y = y1
                    line_val_coords = [x2]

        # Handle last line
        num_vals.append(len(line_val_coords))
        all_val_coords.append(line_val_coords)

        num_years = Counter(num_vals).most_common(1)[0][0]
        if self.debug:
            print(f'NUM YEARS: {num_years}')
            print(f'VAL COORDS: {all_val_coords}')

        # initialize columns based on number of years
        cols = [[] for _ in range(num_years)]

        # add coords to columns if number of coords matches number of years
        for coords in all_val_coords:
            if not len(coords) == num_years:
                continue
            for i, coord in enumerate(coords):
                cols[i].append(coord)

        # get median x value for each column
        self.col_coords = [np.median(col) for col in cols]

        if self.debug:
            print(f'captured column coords: {self.col_coords}')


    def preprocess_text(self, date_x_thresh=300, val_x_thresh=75, y_thresh=50):
        """
        -Converts raw OCR output into structured line objects (RawLine), identifying labels and numeric values.
        -Captures dates and removes junk preceding them.
        """
        assert self.state == State.LOADED_DATA

        output = []  # Final list of RawLine objects

        year_pattern = re.compile(r'(?:19|20)\d{2}')

        # Initialize tracking variables with first OCR line
        start_bbox, first_line, _ = self.ocr_output[0]
        x1, x2, start_y1, start_y2 = get_coords(start_bbox)
        current_x1 = x1
        start_line = first_line
        new_line = RawLine()

        got_years = False
        found_first_year = False
        date_line = None
        # Process each OCR-detected segment starting from the second entry
        for bbox, text, _ in self.ocr_output[1:]:

            # Skip lone characters that are likely noise (except $ or S)
            if len(text) == 1 and text.isalpha() and text not in {'$', 'S'}: #TODO: flagging this --> might mess with parenthese
                if self.debug:
                    print(f'skipping noise: {text}')
                continue
            
            x1, x2, y1, y2 = get_coords(bbox)

            # Captures date and removes lines from output if date captured
            if not got_years:
                if year_pattern.match(text):
                    # Match detected year with closest col
                    distances = [abs(x2 - col_coord) for col_coord in self.col_coords]
                    closest_idx = np.argmin(distances)
                    
                    # confirm captured year is alligned with cols
                    if abs(x2 - self.col_coords[closest_idx]) < date_x_thresh:
                            if not found_first_year:
                                date_line = y1
                                self.years.append(int(text))
                                found_first_year = True
                                if self.debug:
                                    print(f'year found: {text}')
                                continue
                            elif abs(y1 - date_line) < y_thresh: # Confirm subsequent captured years are on the same line
                                self.years.append(int(text))
                                if self.debug:
                                    print(f'year found: {text}')
                                continue
                    else:
                        if self.debug:
                            print(f'failed: year {text} x2: {x2} col: {self.col_coords[closest_idx]}')

                if found_first_year: # If years found already but no more matches, must be done with that line. 
                    got_years = True
                    new_start = len(output) + 1 # make new start the next line of output. Exclude prevous lines
                    
                    if self.debug:
                        print(f'Captured years: {self.years}')

            # Uses vertical allignment to determine whether to append current line or start a new one
            if abs(y1 - start_y1) < y_thresh:
                cleaned_text = text.replace('S', '').replace('$', '').replace(' ', '') # Meant to handle instances where $ of Y2 gets captured with value of Y1 ie '2,019 $'

                # Detect if the text is a financial value
                if detect_vals.match(cleaned_text):

                    # Match val with closest column
                    distances = [abs(x2 - col_x) for col_x in self.col_coords]
                    closest_idx = np.argmin(distances)

                    # Confirms detected val is a val and not cut off from label. Happens when label ends with number
                    if distances[closest_idx] <= val_x_thresh:
                    
                        if self.debug:
                            print(f'DETECTED VAL: {cleaned_text}')

                        # Vals with $ are tracked for openpyxl formatting
                        # Special handling if value ends with dollar sign.
                        if '$' in text or 'S' in text: 
                            if not text.startswith('$') or not text.startswith('S'):  # Prevents $ glued to end of val '2,019 $' from failing horizontal allignment check in build_fs()
                                x2 -= 50 #DOLLAR_SIGN_CORRECTION
                            new_line.add_dollar_sign()
                    
                        # Add value and its right x-bound
                        new_line.add_val_y_val(y1)
                        line_val_coords.append(x2)
                        new_line.add_val(cleaned_text, x2)
                    else:
                        # Val was misaligned and should be part of the label
                        if self.debug:
                            print(f'REJECTED: {text} AT X = {x2}, TOO FAR FROM COL {self.col_coords[closest_idx]}')
                            print('Adding to label instead')
                        start_line += ' ' + text

                else:
                    # $ added in excel formatting. Do not add to RawLine
                    if text.strip() in {'$', 'S'}:
                        new_line.add_dollar_sign()
                        continue
                    
                    else:
                        # Glue OCR text fragments on same line together 
                        if self.debug:
                            print(f'DETECTED LABEL FRAGMENT: {text}')
                        start_line += ' ' + text

            else:
                # Finalize current line
                new_line.add_label(start_line)
                new_line.add_x_coords(current_x1, x2)
                new_line.add_y_vals(start_y1, start_y2)
                if self.debug:
                    print(f'COMPLETED LINE: {start_line}, VALS: {new_line.get_vals()}')
                output.append(new_line)

                # Reset everything for new line
                new_line = RawLine()
                start_y1 = y1
                start_y2 = y2
                line_val_coords = []
                current_x1 = x1
                start_line = text

        # Handle last line
        if start_line:
            new_line.add_label(start_line)
            new_line.add_x_coords(current_x1, x2)
            new_line.add_y_vals(start_y1, start_y2)
            if self.debug:
                print(f'ADDING FINAL LINE: {start_line}')
        output.append(new_line)

        if self.debug:
            print(f'captured dates: {self.years}')

        if self.years is not None and len(self.years) == len(self.col_coords):
            output = output[new_start:]

        # If something went wrong with capturing dates, create default ones
        else:
            self.years = ['Year ' + str(num + 1) for num in range(len(self.col_coords))]
            if self.debug:
                print('Warning. could not find years. Using defaults instead')
        self.raw_lines = output
        self.state = State.PREPROCESSED


    def add_underscore_zeros(self, val_x_thresh=75):
        """
        Inserts '0' values into financial lines where underscores (e.g., "__") were detected in the image.
        """
        assert self.state == State.PREPROCESSED
        can_num = 0  # Candidate counter for debugging

        for (x, y, w, _) in self.underscore_coords:
            can_num += 1
            if self.debug:
                print('\n=== NEW CANDIDATE ===')

            rx = x + w  # Right x-coordinate of underscore (used to match against col_coords)

            for line in self.raw_lines:
                line_y1, line_y2  = line.get_y_vals()
                vals = line.get_vals()

                # Check if underscore y-position falls within this line’s vertical bounds
                if y in range(line_y1, line_y2):

                    # Only add a zero if the number of values on the line is less than number of columns
                    if len(self.col_coords) > len(vals):

                        # Find closest column for underscore
                        candidate_distances = [abs(rx - col_x) for col_x in self.col_coords]
                        candidate_closest_idx = np.argmin(candidate_distances)
                        
                        # Check if the match is within the allowed horizontal threshold
                        if candidate_distances[candidate_closest_idx] <= val_x_thresh:

                            # Get existing column indices for real values in this line
                            existing_col_idxs = [
                                np.argmin([abs(val_coord - col_x) for col_x in self.col_coords])
                                for _, val_coord in vals
                            ]

                            # Check if that column is already filled
                            if candidate_closest_idx in existing_col_idxs:
                                if self.debug:
                                    print(f'SKIPPING candidate {can_num}: column {candidate_closest_idx} already filled. LINE: {line.get_text()}')
                                continue

                            # Safe to insert - add a '0' value for the underscore
                            line.vals.append(('0', rx))
                            if self.debug:
                                print(f'INSERTED candidate {can_num} at col {candidate_closest_idx}, x={rx}, y={y}, LINE: {line.get_text()}')
                            break
                        else:
                            if self.debug:
                                print(f'REJECTING candidate {can_num} with coords x: {rx} y: {y}. Failed x threshold check. LINE: {line.get_text()}')
                            continue
                    else:
                        if self.debug:
                            print(f'REJECTING candidate {can_num} with coords x: {rx} y: {y}. Vals already accounted for. LINE: {line.get_text()}')
                        continue
                else:
                    if self.debug:
                        print(f'REJECTING candidate {can_num} with coords x: {rx} y: {y}. Y thresh failed. LINE: {line.get_text()} LINE Y1: {line_y1} LINE Y2: {line_y2}')


    def assign_summing_lines(self, y_thresh=40):
        """
        Takes summing lines detected by cv2 kernel and assigns them based on y
        position to mark it as a total
        """
        assert self.state == State.PREPROCESSED
        can_num = 0  # Candidate counter for debugging
        found_first = False # Flag to skip any lines before first actual line item found
        top_y = None # First line y val coord. No lines should ever be above here. 

        # get summing line y val
        for (_, y, _, _) in self.summing_line_coords:
            can_num += 1
            if self.debug:
                print(f'\n=== NEW CANDIDATE: {can_num}===')

            for i, line in enumerate(self.raw_lines):
                # confirm line has y vals to compare to summing_line candidate.
                val_y_vals = line.get_val_y_vals()
                if not val_y_vals:
                    continue
                val_y = val_y_vals[0]
                if not found_first:
                    top_y = val_y
                    found_first = True

                # Confirm line y is below first line_item
                if y > top_y:
                    # Confirm line y val and val y val are within threshold
                    if abs(y - val_y) < y_thresh:
                        # Confirm we havent already identified this line as a total
                        if not line.is_total:
                            line.set_total()  # Setting line as total
                            if self.debug:
                                print(f'PLACED Candidate {can_num} with Y: {y} at line {line} with Y: {val_y}')
                            break
                        elif self.debug:
                            print(f'ALREADY PLACED Candidate {can_num} with Y: {y} at line {line} with Y: {val_y}')
            
        if self.debug:
            found_totals = [f'\n{line}' for line in self.raw_lines if line.is_total]
            print('Found totals:')
            for total in found_totals:
                print(f'\n{total}')
                
    def add_indentation(self, x_thresh=50):
        """
        Assigns indentation levels to each RawLine line based on horizontal x-coordinates.

        Args:
            raw_data (list of RawLine): Parsed lines from OCR with bounding box coordinates.
            debug (bool): If True, prints detailed info for debugging.
            x_thresh (int): Threshold for how much x1 must differ to count as a new indent level.
        """

        # Find the smallest x1 as the minimum indentation baseline
        x1, _ = self.raw_lines[0].get_x_coords()
        min_x = x1
        for line in self.raw_lines[1:]:
            x1, _ = line.get_x_coords()
            if x1 < min_x:
                min_x = x1

        # Initialize indentation tracking
        current_indent = 0
        current_baseline = min_x

        # Assign indentaion level to each line
        for line in self.raw_lines:
            x1, _ = line.get_x_coords()
            if self.debug:
                print(f"\nProcessing {line.get_text()}")
                print(f"x1: {x1}, baseline: {current_baseline}, min_x: {min_x}")

            # Case 1: Line is further right than baseline and exceeds threhold so indent += 1
            if x1 - current_baseline > 0 and abs(x1 - current_baseline) > x_thresh:
                current_indent += 1
                current_baseline = x1
                line.add_indentation(current_indent)
                if self.debug:
                    print(f"Case 1: indent += 1 -> {current_indent}")

            # Case 2: Line is roughly aligned with the leftmost margin so reset indent to 0
            elif abs(x1 - min_x) < x_thresh:
                current_indent = 0
                current_baseline = min_x
                line.add_indentation(current_indent)
                if self.debug:
                    print(f"Case 2: indent = 0")
            
            # Case 3: Line is close to current baseline so maintain current indent
            elif abs(x1 - current_baseline) < x_thresh:
                current_baseline = x1
                line.add_indentation(current_indent)
                if self.debug:
                    print(f"Case 3: within threshold, indent stays {current_indent}")

            # Case 4: Line is to the left of current indent baseline and exceeds threshold so indent -= 1
            elif x1 - current_baseline < 0 and abs(x1 - current_baseline) > x_thresh:
                current_indent -= 1
                current_baseline = x1
                line.add_indentation(current_indent)
                if self.debug:
                    print(f"Case 4: indent -= 1 -> {current_indent}")

            # Warning print for edge cases
            else:
                if self.debug:
                    print(f"NO CASE MATCHED FOR LINE: {line}")


    def what_fs(self, score_thresh=5):
        """
        Identifies the type of financial statement (Balance Sheet or Income Statement) from parsed text lines.
        - Iterates through labels of RawLine objects to determine the type of financial statement.
        - Checks if expected words are in labels.
        - If a word is found, adds 1 to the score associated with that statement (ie if BS word is found adds +1 to BS score).
        - The statement type will be determined based on whether more IS words or BS words are detected

        Args:
            cleaned (list of RawLine): A list of parsed text lines from the document, each with `.get_text()`.
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

        for line in self.merged_lines:
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
            if self.debug:
                print(f'BS identified. BS score: {bs_score} IS score: {is_score}')
            return 'BALANCE_SHEET'
        elif is_score > bs_score and is_score >= score_thresh:
            if self.debug:
                print(f'IS identified. IS score: {is_score} BS score: {bs_score}')
            return 'INCOME_STATEMENT'
        else:
            raise ValueError('Could not recognize document')


    def merge_lines(self):
        """
        Merges fragmented line items in financial statements based on formatting cues.
        A merge is performed only if:
        - The current line's label does not start with a capitalized word (likely a continuation).
        - The current line is indented farther than the previous (child of previous).
        - One of the two lines has values, but not both (to avoid overwriting).

        Args:
            lines (list of RawLine): Parsed lines from the OCR output.
            debug (bool): If True, prints detailed merge logic steps.

        Returns:
            list of RawLine: A cleaned list of lines, with appropriate lines merged together.
        """
        assert self.state == State.PREPROCESSED

        output = []

        for line in self.raw_lines:
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
                        if self.debug:
                            print(f'Tried to merge but ran into Error. At least one line must have vals but not both. Lines: {label} | {prev_label}')
                        output.append(line)
                        continue
                    full_label = prev_label.strip() + ' ' + label.strip()
                    if self.debug:
                        print(f'MERGIING LINE. LINE NOW: {full_label}')

                    # if continuation line has data, transfer to line above
                    if vals:
                        prev.vals = vals
                    if d_sign:
                        prev.add_dollar_sign()
                    prev.add_label(full_label)
                else:
                    # failed indentation check
                    if self.debug:
                        print(f'Warning. Line: {label} is not capitalized but could not merge')
                    output.append(line)
            else:
                # Handles normal lines
                output.append(line)

        self.merged_lines = output
        self.state = State.MERGED


    def get_end(self, result):
        """
        Determines Regex to use to identify end of statement
        """
        if result == 'BALANCE_SHEET':
            return re.compile(r'(?i)total.*liabilit(?:y|ies).*equity')
        else:
            return re.compile(r'(?i)^.*net\s+\(?(income|earnings)\)?(?:\s+\(loss\))?.*')
        

    def build_fs(self):
        """
        Builds a FinancialStatement object from parsed OCR line data.
        - Determines the financial statement type (Balance Sheet or Income Statement).
        - Iterates over each parsed line and builds a LineItem object with associated values.
        - Adds headings directly to the financial statement.
        """
        assert self.state == State.MERGED

        end_collection = False
        new_fs = FinancialStatement()

        # Infer financial statement type and stopping point
        fs_type = self.what_fs()
        new_fs.add_type(fs_type)
        end = self.get_end(fs_type)

        for line in self.merged_lines:
            label = line.get_text()
            label = label.strip('.').strip('_')  # for formats where labels and vals separated with periods
            dollar_sign = line.get_dollar_sign()
            indent = line.get_indent()

            new_fs.add_years(self.years)
            new_fs.add_type(fs_type)
            # End detected and line does not match end regex. Exclude useless lines
            if end_collection and not end.match(label):
                if self.debug:
                    print(f'Final line detected. Excluded: {label}')
                break

            new_line_item = LineItem()
            if dollar_sign:
                new_line_item.add_dollar_sign()
            if line.is_total:
                new_line_item.set_total()
            new_line_item.add_indent(indent)


            # Checks if line item with vals
            vals = line.get_vals()
            if not line.get_vals() == []:

                if self.debug:
                    print(f'LINE ITEM DETECTED: {label}')

                # Sets end_collection flag to true. Sometimes multiple lines match end collection and we want the last one which is why the loop does not stop here. 
                if end.match(label) and dollar_sign:
                    if self.debug:
                        print(f'Final line detected and included: {label}')
                    end_collection = True

                # add values to line item
                assigned_vals = [None] * len(self.col_coords)
                if self.debug:
                    print(f'vals before parsing: {vals}')
                for val, val_coord in vals:
                    if detect_vals.match(val):
                        cleaned = val.replace(',', '').replace('_', '').replace('(', '-').replace(')', '')
                        try:
                            val_num = float(cleaned)
                        except ValueError:
                            val_num = 0
                            if self.debug:
                                print(f'COULD NOT PARSE VAL. TREATING AS 0: {val}')
                    else:
                        val_num = 0
                        if self.debug:
                            print(f'VAL NOT RECOGNIZED. TREATING AS 0: {val}')
    
                    # Assign to correct index
                    distances = [abs(val_coord - col_x) for col_x in self.col_coords]
                    closest_idx = np.argmin(distances)
                    assigned_vals[closest_idx] = val_num

                # add data to new line item obj
                new_line_item.add_label(label)
                new_fs.add_line_item(new_line_item)
                # Fill in values for each expected year. Pad with 0s if necessary
                for idx, year in enumerate(self.years):
                    value = assigned_vals[idx] if assigned_vals[idx] is not None else 0
                    new_line_item.add_data(year, value)
                if self.debug:
                    print(f'Completed line item: {new_line_item}')
            else:
                #Must be heading without vals
                if end_collection:  # It may seem like this check isnt necessary but its important esp in IS with "net income per share" lines since they trigger the regex but dont have vals and shouldnt be included since final net income already captured
                    if self.debug:
                        print(f'END REACHED. EXCLUDED: {label}')
                    break

                if self.debug: 
                    print(f'HEADING DETECTED: {label}')
                new_line_item.add_label(label)
                new_fs.add_line_item(new_line_item)

        self.fs = new_fs
        self.state = State.COMPLETED


    def get_summing_ranges(self, off_by_thresh=0):
        """
        Determines indices of line items to add to each subtotal for dynamic summing in final Excel output
        """
        SUBTOTAL = 1 # For formatting in export.py; adds single line to top border of cell
        TOTAL = 2 # For formatting in export.py; adds single line to top and double line to bottom
        REGULAR_VAL = 0 # For formatting in export.py; values is blue with no black borders

        assert self.state == State.COMPLETED

        fs_type = self.fs.fs_type
        tlse = re.compile(r'(?i)total.*liabilit(?:y|ies).*equity')
        ta = re.compile(r'(?i)total assets')

        unaccounted_for = [] # List of indices not used in a sum yet
        year = self.years[0]

        for i, line in enumerate(self.fs.lines):
            # Skip lines without vals
            vals = line.get_data()
            if not vals:
                continue
            val = vals[year]

            # if regulat val, add to unaccounted for
            if not line.is_total:
                unaccounted_for.append((i, val))

            # Total detected! Set summing type and try to find summing range
            else:
                label = line.get_label()
                if not unaccounted_for:
                    print(f'Warning. total detected @ line {label} but all accounted for. Treating as regular val')
                    unaccounted_for.append((i, val))
                    continue

                line.add_summing_type(SUBTOTAL)

                vals = line.get_data()
                target = vals[year]
                
                subsets = get_all_subsets(unaccounted_for)  # Cals fxn in utils to find all subsets of indices in unaccounted for
                solution = find_solution(subsets, target, off_by_thresh) # Finds all possible combos of addition and subtraction of subsets to achieve target val

                #Add summing rnage if solution found
                if solution:
                    line.add_summing_range(solution)
                    if self.debug:
                        print(f'Found solution to line {label}: {solution}')

                    # Update unaccounted for to remove indexes found in solution
                    for i_sol, _ in solution:
                        for item in unaccounted_for[:]:
                            i_u, _ = item
                            if i_sol == i_u:
                                unaccounted_for.remove(item)
                                break
                else:
                    if self.debug:
                        print(f'Warning. Could not find solution for total @ line {line.get_label()}. Treating as regular val')
                    line.add_summing_type(REGULAR_VAL)
                        

                # sets summing type for BS totals; prevents grand totals from being included in unaccounted for
                if fs_type == 'BALANCE_SHEET':
                    if tlse.match(label) or ta.match(label):
                        line.add_summing_type(TOTAL)
                        continue
                # Update unaccounted for to include total
                unaccounted_for.append((i, val))

        # set summing type for net income
        if fs_type == 'INCOME_STATEMENT':
            self.fs.lines[-1].add_summing_type(TOTAL)

        print('Summing ranges:')
        for line in self.fs.lines:
            if line.get_summing_range():
                print(f'Line: {line.get_label()} | Range: {line.get_summing_range()}')


    def debug_output(self, val_x_thresh=75):
        """
        - Draws bboxes and red lines denoting value position on financial statement.
        - Prints text captured in each bbox and its confidence value
        """
        assert self.state == State.PREPROCESSED

        img = np.array(self.img)

        for bbox, text, conf in self.ocr_output:
            if self.debug:
                print(f"{text} | bbox: {bbox} | confidence: {conf:.2f}")
            bbox = [tuple(map(int, point)) for point in bbox]
            cv2.rectangle(img, bbox[0], bbox[2], (0, 255, 0), 2)
            cv2.putText(img, text, bbox[0], cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        if self.col_coords:
            height = img.shape[0]
            for x in self.col_coords:
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


    def export(self, format: Format):
        """
        Exports FinancialStatement object to CSV
        """
        assert self.state == State.COMPLETED

        if(format == Format.CSV):
            export_fs_as_csv(self.fs, self.export_filename)
        elif(format == Format.XLSX):
            export_fs_as_xlsx(self.fs, self.export_filename)
