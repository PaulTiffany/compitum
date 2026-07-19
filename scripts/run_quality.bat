@echo off
setlocal ENABLEDELAYEDEXPANSION
REM Quality gates: ruff + mypy + bandit + pytest
IF NOT EXIST .venv\Scripts\activate.bat (
  echo [run_quality] Missing .venv. Create it and install dev deps: python -m venv .venv ^&^& .venv\Scripts\python -m pip install -U pip ^&^& .venv\Scripts\python -m pip install -e ".[dev]"
  exit /b 1
)

echo [run_quality] Ruff (lint)
.venv\Scripts\ruff check . || goto :error

echo [run_quality] MyPy (types, strict)
.venv\Scripts\mypy --strict --disable-error-code no-any-return src\compitum || goto :error

echo [run_quality] Bandit (security)
.venv\Scripts\bandit -q -r src\compitum -x src\routerbench || goto :error

echo [run_quality] PyTest (unit + property tests)
set HYPOTHESIS_PROFILE=ci
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
set OPENBLAS_NUM_THREADS=1
set NUMEXPR_NUM_THREADS=1
.venv\Scripts\pytest -q -m "not routerbench" || goto :error

echo [run_quality] OK
exit /b 0

:error
echo [run_quality] FAILED (%ERRORLEVEL%)
exit /b %ERRORLEVEL%
