#Requires -Version 7.0
[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$manifest = Get-Content -LiteralPath (Join-Path $projectRoot 'runtimes/manifests/tyrano.json') -Raw | ConvertFrom-Json
$destination = Join-Path $projectRoot $manifest.install_path

if (-not (Get-Command git -ErrorAction SilentlyContinue)) { throw 'Git is required.' }
if (-not (Test-Path -LiteralPath $destination)) {
    & git clone --depth 1 $manifest.repository $destination
    if ($LASTEXITCODE -ne 0) { throw 'TyranoScript clone failed.' }
    $head = (& git -C $destination rev-parse HEAD).Trim()
    if ($head -ne $manifest.commit) {
        & git -C $destination fetch --depth 1 origin $manifest.commit
        if ($LASTEXITCODE -ne 0) { throw 'Cannot fetch the pinned TyranoScript commit.' }
        & git -C $destination checkout --detach $manifest.commit
        if ($LASTEXITCODE -ne 0) { throw 'Cannot check out the pinned TyranoScript commit.' }
    }
}

if (-not (Test-Path -LiteralPath (Join-Path $destination '.git'))) { throw 'TyranoScript must be an independent Git clone.' }
$origin = (& git -C $destination remote get-url origin).Trim()
if ($LASTEXITCODE -ne 0 -or $origin -ne $manifest.repository) { throw 'Unexpected TyranoScript origin; existing files were left unchanged.' }
$head = (& git -C $destination rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $head -ne $manifest.commit) { throw 'TyranoScript commit differs from the manifest; existing files were left unchanged.' }
foreach ($required in @('index.html', 'tyrano/tyrano.js', 'tyrano/plugins/kag/kag.js', 'data/system/Config.tjs', 'LICENCE.txt')) {
    if (-not (Test-Path -LiteralPath (Join-Path $destination $required))) { throw "TyranoScript file is missing: $required" }
}
Write-Host "TyranoScript ready: $destination ($head)"
