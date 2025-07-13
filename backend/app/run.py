import os
from .processors.OcrProcessor import OcrProcessor

if __name__ == "__main__":
    pdf_path = os.path.join("Financials", "BS", "Amazon_bs_20.pdf")  # I change this to test different statements
    ocr = OcrProcessor(pdf_path, debug=False, use_cache=False)
    ocr.process()
    