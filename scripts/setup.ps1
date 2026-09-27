#Requires -Version 7.0
[CmdletBinding()]
param(
    [ValidateSet('all', 'core', 'worker', 'web', 'diffusers', 'irodori', 'llama', 'tyrano')]
    [string[]]$Components = @('all')
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
$all = $Components -contains 'all'
$managedVariables = @('UV_CACHE_DIR', 'UV_PYTHON_INSTALL_DIR', 'UV_PROJECT_ENVIRONMENT', 'UV_PYTHON_PREFERENCE', 'npm_config_cache')
$previousEnvironment = @{}
foreach ($name in $managedVariables) { $previousEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, 'Process') }

function Invoke-Checked {
    param([string]$Program, [string[]]$Arguments)
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Program failed (exit $LASTEXITCODE)" }
}

Push-Location $projectRoot
try {
    $env:UV_CACHE_DIR = Join-Path $projectRoot '.cache\uv'
    $env:UV_PYTHON_PREFERENCE = 'only-managed'
    $env:npm_config_cache = Join-Path $projectRoot '.cache\npm'
    if ($all -or $Components -contains 'core') {
        $env:UV_PYTHON_INSTALL_DIR = Join-Path $projectRoot '.python'
        $env:UV_PROJECT_ENVIRONMENT = Join-Path $projectRoot '.venv'
        Invoke-Checked uv @('python', 'install', '3.12.13', '--no-bin', '--no-registry')
        Invoke-Checked uv @('sync', '--locked', '--python', '3.12.13')
    }
    if ($all -or $Components -contains 'worker') {
        $env:UV_PYTHON_INSTALL_DIR = Join-Path $projectRoot 'services\worker\.python'
        $env:UV_PROJECT_ENVIRONMENT = Join-Path $projectRoot 'services\worker\.venv'
        Invoke-Checked uv @('python', 'install', '3.12.13', '--no-bin', '--no-registry')
        Invoke-Checked uv @('sync', '--locked', '--python', '3.12.13', '--project', 'services/worker')
    }
    if ($all -or $Components -contains 'web') {
        Push-Location (Join-Path $projectRoot 'apps\web')
        try { Invoke-Checked npm.cmd @('ci'); Invoke-Checked npm.cmd @('run', 'build') }
        finally { Pop-Location }
    }
    foreach ($component in @('diffusers', 'irodori', 'llama', 'tyrano')) {
        if ($all -or $Components -contains $component) {
            & (Join-Path $projectRoot "runtimes\installers\setup-$component.ps1")
            if (-not $?) { throw "$component setup failed" }
        }
    }
}
finally {
    Pop-Location
    foreach ($name in $managedVariables) { [Environment]::SetEnvironmentVariable($name, $previousEnvironment[$name], 'Process') }
}


