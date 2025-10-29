@echo off
setlocal
set PYTHONPATH=src
set HYPOTHESIS_PROFILE=ci

if "%1"=="deep" (
  pytest -q -m "invariants and deep"
) else (
  pytest -q tests\invariants
)

endlocal
