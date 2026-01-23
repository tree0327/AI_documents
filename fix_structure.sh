#!/bin/bash
set -x # Print commands as they run

# Base Directory - scripts runs from here
# 1. Cleanup Garbage in 01_파이썬_기초
echo "Cleaning up garbage..."
rm -rf 01_파이썬_기초/crawling
rm -rf 01_파이썬_기초/데이터분석및시각화
rm -rf 01_파이썬_기초/딥러닝
rm -rf 01_파이썬_기초/머신러닝
rm -rf 01_파이썬_기초/수업자료
rm -rf 01_파이썬_기초/예복습모음
rm -rf 01_파이썬_기초/__pycache__

# 2. Create V2 Directory Structure
echo "Creating new directories..."
mkdir -p 01_파이썬/01_기초문법
mkdir -p 01_파이썬/02_자료구조
mkdir -p 01_파이썬/03_객체지향_고급
mkdir -p 02_데이터수집/01_웹크롤링
mkdir -p 03_데이터분석/01_데이터분석_기초
mkdir -p 03_데이터분석/02_데이터_시각화

# 3. Move Python Files
echo "Moving Python files..."
mv "01_파이썬_기초/01_파이썬_시작하기.ipynb" "01_파이썬/01_기초문법/"
mv "01_파이썬_기초/02_Python_Syntax.ipynb" "01_파이썬/01_기초문법/"
mv "01_파이썬_기초/03_Variables.ipynb" "01_파이썬/01_기초문법/"
mv "01_파이썬_기초/04_Strings.ipynb" "01_파이썬/01_기초문법/"
mv "01_파이썬_기초/09_Conditionals.ipynb" "01_파이썬/01_기초문법/"
mv "01_파이썬_기초/10_Loops_For.ipynb" "01_파이썬/01_기초문법/"
mv "01_파이썬_기초/11_Loops_While.ipynb" "01_파이썬/01_기초문법/"
mv "01_파이썬_기초/13_Input_Output.ipynb" "01_파이썬/01_기초문법/"
mv "01_파이썬_기초/14_User_Input.ipynb" "01_파이썬/01_기초문법/"
mv "01_파이썬_기초/16_Functions.ipynb" "01_파이썬/01_기초문법/"
mv "01_파이썬_기초/17_Lambda.ipynb" "01_파이썬/01_기초문법/"

mv "01_파이썬_기초/05_Lists.ipynb" "01_파이썬/02_자료구조/"
mv "01_파이썬_기초/06_Tuples.ipynb" "01_파이썬/02_자료구조/"
mv "01_파이썬_기초/07_Dictionaries.ipynb" "01_파이썬/02_자료구조/"
mv "01_파이썬_기초/08_Sets.ipynb" "01_파이썬/02_자료구조/"
mv "01_파이썬_기초/12_List_Comprehension.ipynb" "01_파이썬/02_자료구조/"

mv "01_파이썬_기초/02_파이썬_중급_붕어빵.ipynb" "01_파이썬/03_객체지향_고급/"
mv "01_파이썬_기초/03_파이썬_고급_선물포장.ipynb" "01_파이썬/03_객체지향_고급/"
mv "01_파이썬_기초/18_Classes.ipynb" "01_파이썬/03_객체지향_고급/"
mv "01_파이썬_기초/19_Modules.ipynb" "01_파이썬/03_객체지향_고급/"
mv "01_파이썬_기초/19module.py" "01_파이썬/03_객체지향_고급/"
mv "01_파이썬_기초/20_Random.ipynb" "01_파이썬/03_객체지향_고급/"
mv "01_파이썬_기초/21_Datetime.ipynb" "01_파이썬/03_객체지향_고급/"

# 4. Move Data Files
echo "Moving Data files..."
# Check if source exists before moving to avoid errors if already moved
if [ -d "02_데이터_정제/01_웹크롤링" ]; then
    cp -r "02_데이터_정제/01_웹크롤링/"* "02_데이터수집/01_웹크롤링/"
fi

if [ -f "03_데이터_분석/01_넘파이_기초_컨테이너.ipynb" ]; then
    mv "03_데이터_분석/01_넘파이_기초_컨테이너.ipynb" "03_데이터분석/01_데이터분석_기초/"
fi
if [ -f "03_데이터_분석/02_판다스_엑셀.ipynb" ]; then
    mv "03_데이터_분석/02_판다스_엑셀.ipynb" "03_데이터분석/01_데이터분석_기초/"
fi
if [ -f "03_데이터_분석/03_시각화_그림그리기.ipynb" ]; then
    mv "03_데이터_분석/03_시각화_그림그리기.ipynb" "03_데이터분석/02_데이터_시각화/"
fi

# 5. Rename Top Levels (if not already renamed)
if [ -d "06_자연어처리" ]; then
    mv "06_자연어처리" "06_자연어처리_NLP"
fi
if [ -d "08_스트림릿" ]; then
    mv "08_스트림릿" "08_웹앱_Streamlit"
fi

# 6. Delete Old Folders
echo "Deleting old folders..."
rm -rf 01_파이썬_기초
rm -rf 02_데이터_정제
rm -rf 03_데이터_분석

echo "Structure Fix Completed!"
