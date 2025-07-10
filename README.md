# To do list:
## TOP PRIORITIES:
- (Andre) OCR fails to detect single charater values (See Apple BS 15). Also can split lines that shouldnt be split if it thinks they are crooked (See Apple BS 21). 
    - Potential fixes: maybe preprocessing to dilate characters so OCR can detect them, download easyocr package and tweak tolerances so splitting does not occur, get a better (hopefully free) ocr
- (Andre) Go through functions and clean them up. Look for ways to make code more efficient
- (Jonas) if case where val was part of label, need to prevent 0s from autofilling
- (Jonas) Continue throwing financial statements at code and see if anything breaks
## REST:
- The order of operations of my code is kind of silly and should be improved. Should probably get y bounds first and then add vals to raw data with bounds check rather than handling 'erroneous_vals' in build_fs. Should also probably do years and end checks right after preprocess text 
- OPTIONAL - Threshold to binarize image set at 160. Worked so far but could cause issues. Alternative is dynamic setting, but in testing led to a ton of artifacts. 
- hook this up to excel and format it with openpyxl
- list of malformed lines that are highlighted in final excel output
- At some point HTML frontend 
- logic that uses y pos to defend against values being split by OCR - would prefer to have better ocr
- OPTIONAL: Add FYE Line

## BS Problems Im aware of:
- Apple 2015: OCR doesnt catch the 0 in current portion of term debt.
- Apple 2015: Date and FYE line too close so they merge and dont get caught by date regex. Tweaking y threshold causes other financial statemnts to break - Likely need to make date detection regex more lenient
- Apple 2021: OCR splits the 26 off of FYE line - not catastrophic since this line isnt used but can forsee this happening to important values so want to avoid this


## IS Problems Im aware of:


