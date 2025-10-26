Param()
Set-Location "$PSScriptRoot\.."
$env:PYTHONPATH = "$PWD;$PWD\src"
if (Test-Path ".venv-routerbench\Scripts\python.exe") {
  & .\.venv-routerbench\Scripts\python.exe -m pytest -q -m routerbench --cov=compitum --cov-branch --cov-append
} else {
  Write-Output ".venv-routerbench not found"
  exit 1
}
