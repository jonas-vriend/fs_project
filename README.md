# To do list:
## TOP PRIORITIES:
- OCR fails to detect single charater values (See UHG BS 24). Also can split lines that shouldnt be split if it thinks they are crooked. 
    - Potential fixes: maybe preprocessing to dilate cjharacters so OCR can detect them, download easyocr package and tweak tolerances so splitting does not occur, get a better (hopefully free) ocr
- Track $ signs since they are actually useful in separating the sections of A and L&SE
- Need to handle tricky _ 0s. IDEA - if the detect_lines finds it, and its within the appropriate x thresholds, and the line item it matches up with has exmpty values, and there isnt a value dorectly above it within a certain threhold then it can be incldued. 
- logic that uses y pos to defend against values being split by OCR
- add indentations
- Need to be able to handle several docs especially overlapping years.
## REST:
- OPTIONAL - Threshold to binarize image set at 160. Worked so far but could cause issues. Alternative is dynamic setting, but in testing led to a ton of artifacts. 
- hook this up to excel and format it with openpyxl
- list of malformed lines that are highlighted in final excel output
- At some point HTML frontend 
- Use some sort of spell check autocorrect on labels to handle OCR hallcuinations - obv would prefer to just have better ocr
- OPTIONAL: Add FYE Line

## BS Problems Im aware of:
- Apple 2015: OCR doesnt catch the 0 in current portion of term debt.
- Apple 2017: Looks good
- Apple 2021: OCR splits the 26 off of FYE line
- Apple 2023: Looks good
- Apple 2024: Looks good
- UHG 2024: single digit value 9 not caught - really bad

## IS Problems Im aware of:
- Walmart 24: multiple lines that say consolidated net income. 
- Walmart 24: values recorded as -- for 0 break regex

