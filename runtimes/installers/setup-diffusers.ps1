[CmdletBinding()]
param([switch]$SkipGpuCheck)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$runtimeRoot = Join-Path $projectRoot 'services\worker\runtimes\diffusers'
$manifest = Get-Content -LiteralPath (Join-Path $projectRoot 'runtimes\manifests\diffusers.json') -Raw | ConvertFrom-Json
$sourceRoot = Join-Path $runtimeRoot 'source'
$pythonExe = Join-Path $runtimeRoot '.venv\Scripts\python.exe'
$managedVariables = @('UV_CACHE_DIR', 'UV_PYTHON_INSTALL_DIR', 'UV_PYTHON_BIN_DIR', 'UV_PYTHON_PREFERENCE', 'UV_PROJECT_ENVIRONMENT', 'UV_LINK_MODE', 'HF_HOME', 'U2NET_HOME', 'NUMBA_CACHE_DIR', 'PYTHONPATH', 'PYTHONNOUSERSITE')
$previousEnvironment = @{}
foreach ($name in $managedVariables) { $previousEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, 'Process') }

function Invoke-CheckedNative {
    param([string]$Command, [string[]]$Arguments)
    & $Command @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Command exited with code $LASTEXITCODE" }
}

try {
    Get-Command uv, git -ErrorAction Stop | Out-Null
    $env:UV_CACHE_DIR = Join-Path $projectRoot '.cache\uv'
    $env:UV_PYTHON_INSTALL_DIR = Join-Path $runtimeRoot '.python'
    $env:UV_PYTHON_BIN_DIR = Join-Path $runtimeRoot '.python-bin'
    $env:UV_PYTHON_PREFERENCE = 'only-managed'
    $env:UV_PROJECT_ENVIRONMENT = Join-Path $runtimeRoot '.venv'
    $env:UV_LINK_MODE = 'copy'
    $env:HF_HOME = Join-Path $runtimeRoot 'cache\huggingface'
    $env:U2NET_HOME = Join-Path $runtimeRoot 'models\rembg'
    $env:NUMBA_CACHE_DIR = Join-Path $runtimeRoot 'cache\numba'
    $env:PYTHONNOUSERSITE = '1'
    Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue

    if (-not (Test-Path -LiteralPath $sourceRoot)) {
        Invoke-CheckedNative -Command git -Arguments @('clone', '--depth', '1', '--no-checkout', $manifest.source_url, $sourceRoot)
        Invoke-CheckedNative -Command git -Arguments @('-C', $sourceRoot, 'fetch', '--depth', '1', 'origin', $manifest.source_commit)
        Invoke-CheckedNative -Command git -Arguments @('-C', $sourceRoot, 'checkout', '--detach', $manifest.source_commit)
    }
    $actualCommit = & git -C $sourceRoot rev-parse HEAD
    if ($LASTEXITCODE -ne 0 -or $actualCommit.Trim() -ne $manifest.source_commit) {
        throw "Diffusers source checkout does not match the pinned commit $($manifest.source_commit): $sourceRoot"
    }
    $sourceChanges = & git -C $sourceRoot status --porcelain
    if ($LASTEXITCODE -ne 0 -or $sourceChanges) { throw "Diffusers source has local changes. Preserve them before reinstalling: $sourceRoot" }

    Invoke-CheckedNative -Command uv -Arguments @('python', 'install', $manifest.python, '--no-config', '--no-bin', '--no-registry')
    Invoke-CheckedNative -Command uv -Arguments @('sync', '--project', $runtimeRoot, '--python', $manifest.python, '--locked')
    $verifyArguments = @('-I', (Join-Path $runtimeRoot 'verify_environment.py'))
    if ($SkipGpuCheck) { $verifyArguments += '--skip-gpu' }
    Invoke-CheckedNative -Command $pythonExe -Arguments $verifyArguments
}
finally {
    foreach ($name in $managedVariables) { [Environment]::SetEnvironmentVariable($name, $previousEnvironment[$name], 'Process') }
}
