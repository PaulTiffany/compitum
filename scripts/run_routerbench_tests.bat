@echo off
CALL .\.venv-routerbench\Scripts\activate.bat
set PYTHONPATH=%PYTHONPATH%;%CD%\src
python -m pytest src\routerbench
