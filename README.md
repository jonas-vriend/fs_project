# To do list:
- Need to test current code against many BS and IS from several companies to identify weaknesses in regex and ocr
- Line removal preprocessing is way too aggressive with financials that have alternating highlighted rows. Half of data gets deleted before OCr even starts 
- Play with other OCRs and see if accuracy improves.
- Need to be able to handle several docs especially overlapping years.
- Defensive programming that identifies likely haluccinations, maintains integrity of line item values, and highlights suspect cells in final output
- clean up functions: build_bs should be renamed to build_fs and split into multiple functions. add comments and docstrings where necessary
- At some point HTML frontend 
- Apple IS doesnt work rn because of different cuts that repeat years and current class construction is inflexible. If issue persists, may need to change attribute from dictionary to something else
- OPTIONAL: Add FYE Line

## BS Problems Im aware of:
- 2015: OCR doesnt catch the 0 in current portion of term debt. Fix: pad vals with 0s if less than expected and add to malformed list for future flagging
- 2017: Minor misspellings from OCR but fine otherwise. 
- 2021: OCR adds extra 3 in other current assets
- 2023: Looks great
- 2024: OCR doesnt catch 7 in AOCI
-Amazon 24: regex misreads commitments and contingencies note as line item and not header Fix: coords check

## IS Problems Im aware of:

