@echo off
setlocal ENABLEDELAYEDEXPANSION
REM One-button validator: local quality + routerbench + artifacts, and optional online dispatch.

set ARG_ONLINE=0
for %%A in (%*) do (
  if /I "%%~A"=="--online" set ARG_ONLINE=1
)

REM Ensure base venv
if not exist .venv\Scripts\python.exe (
  echo [run_all] Creating base venv and installing dev deps
  python -m venv .venv || goto :error
  .venv\Scripts\python -m pip install -U pip || goto :error
  .venv\Scripts\python -m pip install -e ".[dev]" || goto :error
)

echo [run_all] Local quality gates (ruff, mypy strict, bandit, pytest)
call scripts\run_quality.bat || goto :error

REM Prepare RouterBench venv if missing
if not exist .venv-routerbench\Scripts\python.exe (
  echo [run_all] Creating RouterBench venv and installing deps
  python -m venv .venv-routerbench || goto :error
  call .\.venv-routerbench\Scripts\activate.bat
  pip install -U pip || goto :error
  if exist src\routerbench\requirements.txt (
    pip install -r src\routerbench\requirements.txt || goto :error
  ) else (
    echo [run_all] WARNING: src\routerbench\requirements.txt not found. Skipping RB deps.
  )
  pip install -e . || goto :error
  deactivate
)

echo [run_all] Peer-review orchestration (tests, RouterBench if data present, reports)
call scripts\run_peer_review.bat || goto :error

REM Build docs locally (linkcheck + nitpicky HTML)
echo [run_all] Sphinx docs (linkcheck + HTML)
.venv\Scripts\pip install -r docs\requirements.txt || goto :error
.venv\Scripts\sphinx-build -W -b linkcheck docs docs_build\linkcheck || goto :error
.venv\Scripts\sphinx-build -n -W -b html docs docs_build\html || goto :error

REM Optional: quick cosmic-ray run using quick config, summarizing results
if exist cosmic-ray.quick.toml (
  echo [run_all] Cosmic Ray (quick profile)
  del /q session.sqlite 2>nul
  .venv\Scripts\cosmic-ray init --force cosmic-ray.quick.toml session.sqlite || goto :error
  .venv\Scripts\cosmic-ray exec cosmic-ray.quick.toml session.sqlite || goto :error
  .venv\Scripts\cr-report session.sqlite > reports\mutation_summary.txt 2>&1
)

REM Optional: dispatch online CI, Docs, Release (dry-run)
if %ARG_ONLINE%==1 (
  where gh >nul 2>&1
  if errorlevel 1 (
    echo [run_all] gh CLI not found; skipping online dispatch
  ) else (
    echo [run_all] Dispatching online workflows (CI, Docs, Release dry-run)
    gh workflow run CI
    gh workflow run Docs
    gh workflow run Release
  )
)

echo [run_all] COMPLETE
exit /b 0

:error
echo [run_all] FAILED (%ERRORLEVEL%)
exit /b %ERRORLEVEL%

