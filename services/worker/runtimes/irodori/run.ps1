[CmdletBinding()]
param(
    [ValidateSet('infer.py', 'gradio_app.py', 'gradio_app_voicedesign.py', 'verify')]
    [string]$EntryPoint = 'infer.py',
    [switch]$AllowModelDownload,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ScriptArguments
)

$ErrorActionPreference = 'Stop'
$runtimeRoot = $PSScriptRoot
$pythonPath = Join-Path $runtimeRoot '.venv/Scripts/python.exe'
$sourceRoot = Join-Path $runtimeRoot 'source'
$ffmpegExecutables = @(Get-ChildItem -Path (Join-Path $runtimeRoot 'ffmpeg/*/bin/ffmpeg.exe') -File)
if (-not (Test-Path -LiteralPath $pythonPath)) { throw 'Run runtimes/installers/setup-irodori.ps1 first.' }
if ($ffmpegExecutables.Count -ne 1) { throw 'Expected exactly one local FFmpeg shared installation.' }
$runtimeEnvironment = @{
    HF_HOME = (Join-Path $runtimeRoot 'cache/huggingface')
    HF_HUB_CACHE = (Join-Path $runtimeRoot 'cache/huggingface/hub')
    TORCH_HOME = (Join-Path $runtimeRoot 'cache/torch')
    XDG_CACHE_HOME = (Join-Path $runtimeRoot 'cache')
    GRADIO_TEMP_DIR = (Join-Path $runtimeRoot 'cache/gradio')
    PYTHONUTF8 = '1'
    PATH = ($ffmpegExecutables[0].DirectoryName + [IO.Path]::PathSeparator + $env:PATH)
    HF_HUB_OFFLINE = $(if ($AllowModelDownload) { '0' } else { '1' })
    TRANSFORMERS_OFFLINE = $(if ($AllowModelDownload) { '0' } else { '1' })
}
$previousEnvironment = @{}
foreach ($name in $runtimeEnvironment.Keys) {
    $previousEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
    [Environment]::SetEnvironmentVariable($name, $runtimeEnvironment[$name], 'Process')
}
Push-Location -LiteralPath $sourceRoot
try {
    $scriptPath = if ($EntryPoint -eq 'verify') { Join-Path $runtimeRoot 'verify.py' } else { Join-Path $sourceRoot $EntryPoint }
    & $pythonPath -E -s -X utf8 $scriptPath @ScriptArguments
    if ($LASTEXITCODE -ne 0) { throw "Irodori exited with code $LASTEXITCODE." }
}
finally {
    Pop-Location
    foreach ($name in $previousEnvironment.Keys) {
        if ($null -eq $previousEnvironment[$name]) {
            Remove-Item -LiteralPath "Env:$name" -ErrorAction SilentlyContinue
        }
        else {
            [Environment]::SetEnvironmentVariable($name, $previousEnvironment[$name], 'Process')
        }
    }
}
