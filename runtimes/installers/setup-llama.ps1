#Requires -Version 7.0
[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$manifestPath = Join-Path $projectRoot 'runtimes/manifests/llama_cpp.json'
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
$runtimeRoot = Join-Path $projectRoot $manifest.install_path
$sourceRoot = Join-Path $projectRoot $manifest.source_path
$binRoot = Join-Path $runtimeRoot 'bin'
$downloads = Join-Path $runtimeRoot '.downloads'
$licenseRoot = Join-Path $runtimeRoot 'licenses'
$markerPath = Join-Path $runtimeRoot 'installed.json'
$server = Join-Path $projectRoot $manifest.server_path

if (-not (Get-Command git -ErrorAction SilentlyContinue)) { throw 'Git is required.' }
New-Item -ItemType Directory -Force -Path $runtimeRoot | Out-Null
if (-not (Test-Path -LiteralPath $sourceRoot)) {
    & git clone --depth 1 $manifest.repository $sourceRoot
    if ($LASTEXITCODE -ne 0) { throw 'llama.cpp clone failed.' }
    $head = (& git -C $sourceRoot rev-parse HEAD).Trim()
    if ($head -ne $manifest.source_commit) {
        & git -C $sourceRoot fetch --depth 1 origin $manifest.source_commit
        if ($LASTEXITCODE -ne 0) { throw 'Cannot fetch the pinned llama.cpp source commit.' }
        & git -C $sourceRoot checkout --detach $manifest.source_commit
        if ($LASTEXITCODE -ne 0) { throw 'Cannot check out the pinned llama.cpp source commit.' }
    }
}
if (-not (Test-Path -LiteralPath (Join-Path $sourceRoot '.git'))) { throw 'llama.cpp source must be an independent Git clone.' }
$origin = (& git -C $sourceRoot remote get-url origin).Trim()
if ($LASTEXITCODE -ne 0 -or $origin -ne $manifest.repository) { throw 'Unexpected llama.cpp origin; source was left unchanged.' }
$head = (& git -C $sourceRoot rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $head -ne $manifest.source_commit) { throw 'llama.cpp source differs from the pinned commit; source was left unchanged.' }

$manifestHash = (Get-FileHash -LiteralPath $manifestPath -Algorithm SHA256).Hash.ToLowerInvariant()
$alreadyInstalled = $false
if (Test-Path -LiteralPath $markerPath) {
    $installed = Get-Content -LiteralPath $markerPath -Raw | ConvertFrom-Json
    if ($installed.manifest_sha256 -ne $manifestHash) { throw 'Installed llama.cpp version differs from the manifest. Use a separate runtime directory for an upgrade.' }
    $alreadyInstalled = $true
}

if (-not $alreadyInstalled) {
    New-Item -ItemType Directory -Force -Path $binRoot, $downloads, $licenseRoot | Out-Null
    foreach ($asset in $manifest.assets) {
        $archive = Join-Path $downloads $asset.name
        $valid = (Test-Path -LiteralPath $archive) -and ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash -eq $asset.sha256)
        if (-not $valid) {
            Write-Host "Downloading $($asset.name)"
            Invoke-WebRequest -Uri $asset.url -OutFile $archive
        }
        if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash -ne $asset.sha256) { throw "SHA-256 mismatch: $($asset.name)" }
        Expand-Archive -LiteralPath $archive -DestinationPath $binRoot -Force
    }
    $license = Join-Path $licenseRoot 'llama.cpp-LICENSE'
    if (-not (Test-Path -LiteralPath $license) -or (Get-FileHash -LiteralPath $license -Algorithm SHA256).Hash -ne $manifest.license_sha256) {
        Invoke-WebRequest -Uri $manifest.license_url -OutFile $license
    }
    if ((Get-FileHash -LiteralPath $license -Algorithm SHA256).Hash -ne $manifest.license_sha256) { throw 'llama.cpp license SHA-256 mismatch.' }
}

foreach ($required in @($server, (Join-Path $binRoot 'ggml-cuda.dll'), (Join-Path $licenseRoot 'llama.cpp-LICENSE'))) {
    if (-not (Test-Path -LiteralPath $required)) { throw "Installed runtime file is missing: $required" }
}
$version = (& $server --version 2>&1 | Out-String).Trim()
if ($LASTEXITCODE -ne 0) { throw "llama-server --version failed: $version" }
$expectedBuild = $manifest.binary_release.TrimStart('b')
if ($version -notmatch "(?:version:\s+|build\s+)$expectedBuild\b" -or $version -notmatch [regex]::Escape($manifest.binary_commit.Substring(0, 9))) {
    throw "Unexpected llama.cpp version: $version"
}
$devices = (& $server --list-devices 2>&1 | Out-String).Trim()
if ($LASTEXITCODE -ne 0 -or $devices -notmatch 'CUDA\d') { throw "CUDA GPU detection failed: $devices" }
Write-Host $version
Write-Host $devices

@{
    manifest_sha256 = $manifestHash
    source_commit = $manifest.source_commit
    binary_release = $manifest.binary_release
    binary_commit = $manifest.binary_commit
    checked_at_utc = [DateTime]::UtcNow.ToString('o')
    version = $version
    devices = $devices
} | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $markerPath -Encoding utf8
Write-Host "llama.cpp ready: $server"
