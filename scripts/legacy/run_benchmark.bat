@echo off
set COMPITUM_SKIP_PERF_ASSERTIONS=1
set COMPITUM_STEPS=400
set COMPITUM_REFIT_POLICY=never
set COMPITUM_UPDATE_BATCH_SIZE=100000
.venv\Scripts\pytest.exe -q --benchmark-min-time=0.01