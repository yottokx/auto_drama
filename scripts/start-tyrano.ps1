[CmdletBinding()]
param([int]$Port = 8080)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
$player = Join-Path $projectRoot 'tyranoscript'
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath (Join-Path $player 'index.html'))) { throw 'Tyrano is not installed.' }
& $python -I -m http.server $Port --bind 127.0.0.1 --directory $player
if ($LASTEXITCODE -ne 0) { throw 'Tyrano static server exited with an error.' }

