# To do list:
- Need to test current code against many BS from several companies to identify weaknesses in regex and ocr
- Need to be able to handle several docs. 
- Needs to be able to handle financial statements that are more than 1 page long. 
- Make a bank of common regex inaccuracies to clean labels ie $ misread as S
- OPTIONAL: Add FYE Line
- At some point HTML frontend 
- Play with other OCRs and see if accuracy improves.
- Defensive programming that identifies likely haluccinations, maintains integrity of line item values, and highlights suspect cells in final output
- clean up functions: build_bs should be renamed to build_fs and split into multiple functions. add comments and docstrings where necessary

# BS Problems Im aware of:
- 2015: OCR doesnt catch the 0 in current portion of term debt. Fix: pad vals with 0s if less than expected and add to malformed list for future flagging
-2017: Minor misspellings but fine otherwise. 
-2021: OCR adds extra 3 in other current assets
-2023: Looks great
-2024: OCR doesnt catch 7 in AOCI

# IS Problems Im aware of: