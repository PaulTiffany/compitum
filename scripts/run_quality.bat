@echo off
setlocal ENABLEDELAYEDEXPANSION
REM Quality gates: ruff + mypy + bandit + pytest
IF NOT EXIST .venv\Scripts\activate.bat (
  echo [run_quality] Missing .venv. Create it and install dev deps: python -m venv .venv ^&^& .venv\Scripts\python -m pip install -U pip ^&^& .venv\Scripts\python -m pip install -e ".[dev]"
  exit /b 1
)

echo [run_quality] Ruff (lint)
.venv\Scripts\ruff check . || goto :error

echo [run_quality] MyPy (types)
.venv\Scripts\mypy -p compitum --ignore-missing-imports --hide-error-context || goto :error

echo [run_quality] Bandit (security)
.venv\Scripts\bandit -q -r src\compitum -x src\routerbench || goto :error

echo [run_quality] PyTest (unit + property tests)
set HYPOTHESIS_PROFILE=ci
.venv\Scripts\pytest -q || goto :error

echo [run_quality] OK
exit /b 0

:error
echo [run_quality] FAILED (%ERRORLEVEL%)
exit /b %ERRORLEVEL%

