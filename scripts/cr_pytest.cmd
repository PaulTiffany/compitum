@echo off
setlocal ENABLEDELAYEDEXPANSION

REM Hypothesis settings (defaults for mutation testing)
if "%HYPOTHESIS_PROFILE%"=="" set HYPOTHESIS_PROFILE=mutation
if "%HYPOTHESIS_SEED%"=="" set HYPOTHESIS_SEED=1

REM Threading for determinism
if "%OMP_NUM_THREADS%"=="" set OMP_NUM_THREADS=1
if "%MKL_NUM_THREADS%"=="" set MKL_NUM_THREADS=1
if "%OPENBLAS_NUM_THREADS%"=="" set OPENBLAS_NUM_THREADS=1
if "%NUMEXPR_NUM_THREADS%"=="" set NUMEXPR_NUM_THREADS=1

REM Optional coverage to guide mutation tools
if "%PYTEST_COVERAGE%"=="1" (
  pytest -q --cov=compitum --cov-branch --cov-report=term-missing
) else (
  pytest -q
)

endlocal