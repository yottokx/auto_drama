[CmdletBinding()]
param(
    [string]$Coordinator = 'http://127.0.0.1:8000',
    [string]$Name = 'local',
    [double]$PollInterval = 2,
    [switch]$Once,
    [switch]$ExportOnly,
    [switch]$NoVoiceReuse,
    [switch]$NoImageReuse,
    [switch]$NoMusicReuse,
    [string]$WorkDir
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
$python = Join-Path $projectRoot 'services\worker\.venv\Scripts\python.exe'
$entrypoint = Join-Path $projectRoot 'services\worker\__main__.py'
if (-not (Test-Path -LiteralPath $python)) { throw 'Run scripts/setup.ps1 -Components worker first.' }
$workerArguments = @('-I', '-u', '-X', 'utf8', $entrypoint, '--coordinator', $Coordinator, '--name', $Name,
    '--poll-interval', $PollInterval.ToString([System.Globalization.CultureInfo]::InvariantCulture))
if ($Once) { $workerArguments += '--once' }
if ($ExportOnly) { $workerArguments += '--export-only' }
if ($NoVoiceReuse) { $workerArguments += '--no-voice-reuse' }
if ($NoImageReuse) { $workerArguments += '--no-image-reuse' }
if ($NoMusicReuse) { $workerArguments += '--no-music-reuse' }
if ($WorkDir) { $workerArguments += @('--work-dir', $WorkDir) }
Push-Location $projectRoot
try {
    & $python @workerArguments
    if ($LASTEXITCODE -ne 0) { throw 'Worker exited with an error.' }
}
finally { Pop-Location }
