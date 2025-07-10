# To do list:
## TOP PRIORITIES:
- (Andre) OCR fails to detect single charater values (See Apple BS 15). Also can split lines that shouldnt be split if it thinks they are crooked (See Apple BS 21). 
    - Potential fixes: maybe preprocessing to dilate characters so OCR can detect them, download easyocr package and tweak tolerances so splitting does not occur, get a better (hopefully free) ocr
- (Andre) Go through functions and clean them up. Look for ways to make code more efficient
- (Jonas) if case where val was part of label, need to prevent 0s from autofilling
- (Jonas) Continue throwing financial statements at code and see if anything breaks
- (Jonas) Look into improving extract date logic
- (Jonas or Andre) Need to split preprocess text into two functions. Should identify val columns first and then assign stuff to label and val. Minimizes potential bugs
## REST:
- OPTIONAL - Threshold to binarize image set at 160. Worked so far but could cause issues. Alternative is dynamic setting, but in testing led to a ton of artifacts. 
- BH balance sheet is split into two. Consider supporting multi page bs
- OPTIONAL - Consider change line item from dict in case multiple cuts (Apple IS)? 
- hook this up to excel and format it with openpyxl
- list of malformed lines that are highlighted in final excel output
- At some point HTML frontend 
- logic that uses y pos to defend against values being split by OCR - would prefer to have better ocr
- OPTIONAL: Add FYE Line

## BS Problems Im aware of:
- Apple 2015: OCR doesnt catch the 0 in current portion of term debt.
- Apple 2015: Date and FYE line too close so they merge and dont get caught by date regex. Tweaking y threshold causes other financial statemnts to break - Likely need to make date detection regex more lenient
- Apple 2021: OCR splits the 26 off of FYE line - not catastrophic since this line isnt used but can forsee this happening to important values so want to avoid this
- GE 18: Misses single digit preferred stock vals: 6, 6
- GE18-24 cant find years
- GE 20 - underscore 0s logic struggling with densely packed text
- Accenture 24 - 2 places with dates. Obvious solution would be breaking up build_fs into two functions. finding year should factor in column alignment
## IS Problems Im aware of:
- UHG 2018: Really bad hallcuination where $ treated as 8. More evidence that I should probably split preprocess_text into two functions 

