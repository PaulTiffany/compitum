@echo off
CALL .\.venv-routerbench\Scripts\activate.bat
set PYTHONPATH=%PYTHONPATH%;%CD%\src
REM Try upstream module first; fall back to a bounded local runner if optional deps are missing.
python -m tools.run_routerbench_entry %*
