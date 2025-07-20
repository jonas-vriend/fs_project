# To do list:
## TOP PRIORITIES:
- (Andre) OCR fails to detect single charater values (See Apple BS 15). Also can split lines that shouldnt be split if it thinks they are crooked (See Apple BS 21). 
    - Potential fixes: maybe preprocessing to dilate characters so OCR can detect them, download easyocr package and tweak tolerances so splitting does not occur, get a better (hopefully free) ocr
- (Jonas) Work on summing_line function

## REST:
- OPTIONAL - Threshold to binarize image set at 160. Worked so far but could cause issues. Alternative is dynamic setting, but in testing led to a ton of artifacts. 
- (Jonas) Continue throwing financial statements at code and see if anything breaks
- BH balance sheet is split into two. Consider supporting multi page bs
- list of malformed lines that are highlighted in final excel output
- At some point HTML frontend 
- logic that uses y pos to defend against values being split by OCR - would prefer to have better ocr

## Problems Im aware of:
### OCR ERRORS
- Apple BS 2015: Doesnt catch the 0 in current portion of term debt.
- Apple BS 2021: Splits the 26 off of FYE line - not catastrophic since this line isnt used but can forsee this happening to important values so want to avoid this
- Amazon BS 22 - Total current assets split into 2 bboxes. evals to 0 
- GE BS  18: Misses single digit preferred stock vals: 6, 6
- UHG BS 24 - single digit 9s not captured in common stock line
- UHG IS 2018: Really bad hallcuination where $ treated as 8.
- Hershey BS 24: Accounts receiveable split in two
### LOGIC ERRORS
- GE BS 20 - underscore 0s logic struggling with densely packed text
- GE BS 22 - merged junk to final line
- BH BS 24: auditors are idiots and didnt label the summing line. Maybe detect if a line has an empty label and default to ''
- HAL BS 24: summing range logic breaks because 'Company Shareholder's equity' is a summing line without total in it. TBH i think this is the auditors fault. Not sure this is actionable

## Starting Development

We need to install the pyenv package, which is a Python Version Manager. If not used, you can run into errors because some packages are dependent on specific of python.

The following install pyenv and then sets the version of python in this directory/project to 3.10.13. 
```
brew update
brew install pyenv

echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zprofile
echo 'eval "$(pyenv init --path)"' >> ~/.zprofile
echo 'eval "$(pyenv init -)"' >> ~/.zshrc
exec "$SHELL"

pyenv install 3.10.13
pyenv local 3.10.13
```

Next, we want to create a virtual environment to create an isolated ecosystem for this packages to exist (and won't have dependecy issues with downloads outside this directory).

```
python3 -m venv env
source env/bin/activate
```

Install dependencies/packages.
```
brew install poppler
pip install -r requirements.txt
```
Note: Ensure you are in the latest version of Python

## Run package
Ensure you are in the fs_project outermost directory.

Ensure you are in your virtual environment - you should see `(env)` in your terminal. If you are not in your virtual environment, you must: `source env/bin/activate`.

To execute the run script:
```
python3 -m backend.app.run
```
Note: the `-m` is significant because it allows the backend.app directory to me recognized as a module/package. Without this, fs_app would be seen as a stand-alone scripy and the important would break. 

Note: `run.py` is where you will run anything in the backend. Modify this function with the desired tasks and execute with the command above. 
