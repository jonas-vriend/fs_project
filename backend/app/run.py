import os
from .processors.OcrProcessor import OcrProcessor
from .models import Format

if __name__ == "__main__":
    pdf_path = os.path.join("Financials", "BS", "Amazon_bs_20.pdf")  # I change this to test different statements
    ocr = OcrProcessor(pdf_path, debug=False, use_cache=False, export_filename="financial_statement")
    ocr.process()
    ocr.export(Format.CSV)
    ocr.export(Format.XLSX)
    