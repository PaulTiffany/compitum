#!/usr/bin/env pwsh
param(
  [string]$Config = "cosmic-ray.windows.toml",
  [string]$Session = "cr_session.sqlite",
  [switch]$Coverage
)

$ErrorActionPreference = 'Stop'

# Ensure cosmic-ray installed
python -m pip install --upgrade pip > $null
python -m pip install cosmic-ray cr-report > $null

# Optionally enable coverage guiding for tests
if ($Coverage) { $env:PYTEST_COVERAGE = '1' }

# Initialize and execute
cosmic-ray init --force $Config $Session
cosmic-ray exec $Config $Session

# Reports
cr-report $Session | Tee-Object -FilePath reports\cr_summary_windows.txt
cosmic-ray dump $Session > reports\cr_dump_windows.jsonl