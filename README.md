# To do list:
- Need to test current code against many BS and IS from several companies to identify weaknesses in regex and ocr
- OPTIONAL - Threshold to binarize image set at 160. Worked so far but could cause issues. Alternative is dynamic setting, but in texting led to a ton of artifacts. 
- Play with other OCRs and see if accuracy improves.
- Need to be able to handle several docs especially overlapping years.
- Defensive programming that identifies likely haluccinations, maintains integrity of line item values, and highlights suspect cells in final output
- clean up functions: build_bs should be renamed to build_fs and split into multiple functions. add comments and docstrings where necessary
- At some point HTML frontend 
- Apple IS doesnt work rn because of different cuts that repeat years and current class construction is inflexible. If issue persists, may need to change attribute from dictionary to something else
- OPTIONAL: Add FYE Line

## BS Problems Im aware of:
- Apple 2015: OCR doesnt catch the 0 in current portion of term debt.
- Apple 2017: Minor misspellings from OCR but fine otherwise. 
- Apple 2021: Looks great
- Apple 2023: Looks great
- Apple 2024: OCR doesnt catch 7 in AOCI


## IS Problems Im aware of:
- Walmart 24: multiple lines that say consolidated net income. 
- Walmart 24: values recorded as -- for 0 break regex

## Redesign:
- consider altering OCR function to allow for more lenient threshold for joining lines.

TO DO TOMORROW
- At some point probabky need to store text as dict and include x coords for indentation / formatting 
- Make found date logic more flexible so that it adds junk and doesnt break if the date isnt found but deletes it if the date is found. 
