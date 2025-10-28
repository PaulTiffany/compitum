@echo off
setlocal enabledelayedexpansion

REM Simple wrapper to fetch the RouterBench 5-shot pickle
set PY=%~dp0..\.venv\Scripts\python.exe
if exist "%PY%" (
  "%PY%" "%~dp0fetch_routerbench.py" %*
  goto :eof
) else (
  python "%~dp0fetch_routerbench.py" %*
)

endlocal
