@echo off
setlocal ENABLEDELAYEDEXPANSION
REM Peer-review orchestrator: assumes .venv-routerbench exists with RB deps

IF NOT EXIST .venv-routerbench\Scripts\activate.bat (
  echo [peer_review] Missing .venv-routerbench. Create and install per README.md (RouterBench section).
  exit /b 1
)

set ROOT=%CD%
set PYTHONPATH=%ROOT%;%ROOT%\src

call .\.venv-routerbench\Scripts\activate.bat

REM Build consolidated report from existing or newly generated artifacts
set CONFIG=data\rb_clean\evaluate_routers.yaml
set REPORT=reports\report_release.html

echo [peer_review] Orchestrating with tools\ci_orchestrator.py
python tools\ci_orchestrator.py --all --config %CONFIG% --report-out %REPORT%
set EXITCODE=%ERRORLEVEL%

deactivate
exit /b %EXITCODE%

