@echo off
setlocal

python demo.py --conf-thres 1
python demo.py --conf-thres 1 --source data/example01.jpg
python demo.py --conf-thres 1 --source data/example02.jpg
python demo.py --conf-thres 1 --source data/example03.jpg
python demo.py --conf-thres 1 --source data/example04.jpg
python demo.py --conf-thres 1 --source data/example05.jpg

pause
endlocal