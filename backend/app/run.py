import os
from .processors.OcrProcessor import OcrProcessor
from .models import Format

if __name__ == "__main__":
    pdf_path = os.path.join("Financials", "BS", "Accenture_bs_18.pdf")  # I change this to test different statements
    
    handler = OcrProcessor(pdf_path, debug=True, use_cache=True, export_filename="financial_statement") # initialize OcrProcessor objcet
    
    handler.process() # calls all the OCR functions and generate FS object
    handler.export(Format.CSV) # export FS object to CSV
    handler.export(Format.XLSX) # export FS object to XLSX
    