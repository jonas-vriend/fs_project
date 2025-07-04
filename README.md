# To do list:
## TOP PRIORITIES:
- (Andre) OCR fails to detect single charater values (See UHG BS 24). Also can split lines that shouldnt be split if it thinks they are crooked. 
    - Potential fixes: maybe preprocessing to dilate characters so OCR can detect them, download easyocr package and tweak tolerances so splitting does not occur, get a better (hopefully free) ocr
- (Jonas or Andre) Need to handle tricky _ 0s. IDEA - if the detect_lines finds it, and its within the appropriate x thresholds, and the line item it matches up with has exmpty values, and there isnt a value dorectly above it within a certain threhold then it can be incldued. 
- (Jonas or Andre) Need to be able to handle batch ingestion of several docs especially with overlapping years.
## REST:
- Go through functions and clean them up. Look for ways to make code more efficient
- OPTIONAL - Threshold to binarize image set at 160. Worked so far but could cause issues. Alternative is dynamic setting, but in testing led to a ton of artifacts. 
- hook this up to excel and format it with openpyxl
- list of malformed lines that are highlighted in final excel output
- At some point HTML frontend 
- logic that uses y pos to defend against values being split by OCR - would prefer to have better ocr
- Use some sort of spell check autocorrect on labels to handle OCR hallcuinations - obv would prefer to have better ocr
- OPTIONAL: Add FYE Line

## BS Problems Im aware of:
- Apple 2015: OCR doesnt catch the 0 in current portion of term debt.
- Apple 2017: Looks good
- Apple 2021: OCR splits the 26 off of FYE line - not catastrophic since this line isnt used but can forsee this happening to important values so want to avoid this
- Apple 2023: Looks good
- Apple 2024: Looks good
- UHG 2024: single digit value 9 not caught - really bad

## IS Problems Im aware of:
- Walmart 24: values recorded as -- for 0
- UHG 22: Net income was split into 2 lines. WTF

