$ErrorActionPreference = 'Stop'
$ProjectPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $ProjectPython)) { throw 'Create the project .venv first; see README.md.' }
& $ProjectPython (Join-Path $PSScriptRoot 'tools\install_minepy.py')
if ($LASTEXITCODE -ne 0) { throw 'minepy build failed. Read the first compiler error above; see README.md.' }
