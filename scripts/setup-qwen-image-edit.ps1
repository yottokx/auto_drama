[CmdletBinding()]
param()
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
$python = Join-Path $projectRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw 'Project Python is missing. Run scripts/setup.ps1 first.' }
& $python -u -X utf8 (Join-Path $PSScriptRoot 'image_edit/setup_runtime.py')
if ($LASTEXITCODE -ne 0) { throw 'Qwen Image 2.1 runtime setup failed. See the log above.' }
