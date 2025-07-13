# import openpyxl as xl
# import openpyxl.styles as style
# import openpyxl.utils as utils
# import os
# from .processors.OcrProcessor import OcrProcessor

# # Define shared styles
# BLACK_FILL = style.PatternFill(fill_type="solid", start_color="FF000000", end_color="FF000000")
# YELLOW_FILL = style.PatternFill(fill_type="solid", start_color="FFFFFF00", end_color="FFFFFF00")
# HEADER_FONT = style.Font(name="Helvetica Neue", size=10, color="FFFFFF", bold=True)
# BLUE_FONT = style.Font(name="Helvetica Neue", size=8, color="0070c0")
# RED_ITALIC_FONT = style.Font(name="Helvetica Neue", size=8, color="FF0000", italic=True)
# LABEL_FONT = style.Font(name="Helvetica Neue", size=8)


# if __name__ == "__main__":
#     pdf_path = os.path.join("Financials", "BS", "Amazon_bs_20.pdf")  # I change this to test different statements
#     ocr = OcrProcessor(pdf_path, debug=False, use_cache=False)
#     ocr.process()
    

# # copying main from fs_app in this to access fs for debugging purposes. Will delete later
# def run_fs_app(debug=False, use_cache=True):
#     """
#     Orchestrates the pipeline:
#     - Load/caches OCR
#     - Extracts structure and values
#     - Builds financial statement object
#     """
#     ocr_output, processed_img, underscore_coords = fs.get_data(use_cache, debug)
#     col_coords, lines = fs.preprocess_text(ocr_output, debug)
#     if underscore_coords:
#         fs.add_underscore_zeros(lines, col_coords, underscore_coords, debug)
#     fs.debug_output(ocr_output, processed_img, col_coords, val_x_thresh=75, debug=False)
#     fs.add_indentation(lines)
#     merged_lines = fs.merge_lines(lines, debug)
#     completed = fs.build_fs(col_coords, merged_lines, debug)
#     return completed

# new_fs = run_fs_app()
# lines = new_fs.get_lines()
# years = new_fs.get_years()

# wb = xl.Workbook()
# del wb['Sheet']
# sheet_name = new_fs.get_type()
# ws_new = wb.create_sheet(title=sheet_name)

# def format_line_items(line_items, start_row=4, start_col=2):
#     for i, line in enumerate(line_items):
#         row = start_row + i
#         label, values, dollar_sign, indent_level = line.get_all()

#         # Add label cell
#         cell = ws_new.cell(row=row, column=start_col, value=label)
#         cell.font = LABEL_FONT

#         # Set alignment: center if all caps heading, else indent
#         if label.isupper() and label.replace(" ", "").isalpha():
#             cell.alignment = style.Alignment(horizontal="center", indent=0)
#         else:
#             cell.alignment = style.Alignment(horizontal="left", indent=indent_level)

#         # Add values
#         for j, year in enumerate(years):
#             val = values.get(year, '')
#             col = start_col + 1 + j
#             val_cell = ws_new.cell(row=row, column=col, value=val)
#             val_cell.font = BLUE_FONT

#             if isinstance(val, int):
#                 if dollar_sign:
#                     val_cell.number_format = '_("$"* #,##0_);_("$"* (#,##0)'
#                 else:
#                     val_cell.number_format = '#,##0;(#,##0)'
#             else:
#                 val_cell.number_format = 'General'
    
#     # Add balance check
#     row += 2
#     balance_check = ws_new.cell(row=row, column=start_col, value='Balance Check')
#     balance_check.font = RED_ITALIC_FONT

#     for i, year in enumerate(years):
#         col = start_col + 1 + i
#         col_letter = utils.get_column_letter(col)
#         formula = f"={col_letter}35 - {col_letter}15"  # Replace later with dynamic logic
#         balance_val = ws_new.cell(row=row, column=col)
#         balance_val.value = formula
#         balance_val.font = RED_ITALIC_FONT

# def build_header(fs, start_row=1, start_col=2):
#     # Fill black background
#     for i in range(3):
#         for j in range(len(years) + 1):
#             row = start_row + i
#             col = start_col + j
#             cell = ws_new.cell(row=row, column=col)
#             cell.fill = BLACK_FILL

#         # Add company name in yellow
#     company_name = ws_new.cell(row=start_row, column=start_col, value='COMPANY NAME')
#     company_name.fill = YELLOW_FILL
#     company_name.font = style.Font(name="Helvetica Neue", size=24, underline="single")

#     # Add Financial Statement type
#     row = start_row + 1
#     fs_type = fs.get_type()
#     label = 'Consolidated Balance Sheets' if fs_type == 'BALANCE_SHEET' else 'Consolidated Income Statements'
#     type_cell = ws_new.cell(row=row, column=start_col, value=label)
#     type_cell.font = HEADER_FONT

#     # Add Years
#     for i, year in enumerate(years):
#         row = start_row + 2
#         col = start_col + 1 + i
#         year_cell = ws_new.cell(row=row, column=col, value=year)
#         year_cell.alignment = style.Alignment(horizontal="center")
#         year_cell.font = HEADER_FONT

# # Run formatting
# format_line_items(lines)
# build_header(new_fs)

# # Save to file
# output_file = "formatted_output.xlsx"
# wb.save(output_file)
# print(f"Excel file saved as: {output_file}")