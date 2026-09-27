[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$manifest = Get-Content -LiteralPath (Join-Path $projectRoot 'runtimes/manifests/irodori.json') -Raw | ConvertFrom-Json
$runtimeRoot = Join-Path $projectRoot $manifest.runtime_directory
$sourceRoot = Join-Path $projectRoot $manifest.source_directory

foreach ($command in @('git', 'uv', 'tar')) {
    if (-not (Get-Command $command -ErrorAction SilentlyContinue)) { throw "Required command not found: $command" }
}
New-Item -ItemType Directory -Force -Path $runtimeRoot | Out-Null
if (-not (Test-Path -LiteralPath $sourceRoot)) {
    & git clone --no-checkout $manifest.repository $sourceRoot
    if ($LASTEXITCODE -ne 0) { throw 'Irodori git clone failed.' }
    & git -C $sourceRoot checkout --detach $manifest.commit
    if ($LASTEXITCODE -ne 0) { throw 'Irodori checkout failed.' }
}
if (-not (Test-Path -LiteralPath (Join-Path $sourceRoot '.git'))) { throw 'Irodori source must be a Git clone.' }
$actualCommit = & git -C $sourceRoot rev-parse HEAD
if ($LASTEXITCODE -ne 0 -or $actualCommit.Trim() -ne $manifest.commit) { throw 'Irodori commit differs from the manifest. Keep local changes and review the manifest before updating.' }
$actualRemote = & git -C $sourceRoot remote get-url origin
if ($LASTEXITCODE -ne 0 -or $actualRemote.Trim() -ne $manifest.repository) { throw 'Unexpected Irodori origin URL.' }
$localChanges = & git -C $sourceRoot status --porcelain --untracked-files=no
if ($LASTEXITCODE -ne 0 -or $localChanges) { throw 'Irodori tracked files have local changes; refusing to overwrite them.' }
$actualLock = & git -C $sourceRoot rev-parse 'HEAD:uv.lock'
if ($LASTEXITCODE -ne 0 -or $actualLock.Trim() -ne $manifest.upstream_lock_git_blob) { throw 'Unexpected Irodori dependency lock.' }

$runtimeEnvironment = @{
    UV_CACHE_DIR = (Join-Path $projectRoot '.cache/uv')
    UV_PYTHON_INSTALL_DIR = (Join-Path $runtimeRoot '.python')
    UV_PYTHON_BIN_DIR = (Join-Path $runtimeRoot '.python/bin')
    UV_PROJECT_ENVIRONMENT = (Join-Path $runtimeRoot '.venv')
    UV_PYTHON_PREFERENCE = 'only-managed'
    UV_LINK_MODE = 'copy'
}
$previousEnvironment = @{}
foreach ($name in $runtimeEnvironment.Keys) {
    $previousEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
    [Environment]::SetEnvironmentVariable($name, $runtimeEnvironment[$name], 'Process')
}
try {
    & uv python install $manifest.python_version --no-bin --no-registry
    if ($LASTEXITCODE -ne 0) { throw 'Irodori Python install failed.' }
    & uv sync --project $sourceRoot --locked --extra $manifest.uv_extra --python $manifest.python_version
    if ($LASTEXITCODE -ne 0) { throw 'Irodori dependency install failed.' }

    $ffmpegRoot = Join-Path $runtimeRoot $manifest.ffmpeg.directory
    if (-not (Test-Path -LiteralPath (Join-Path $ffmpegRoot 'bin/ffmpeg.exe'))) {
        $downloadsRoot = Join-Path $runtimeRoot 'downloads'
        New-Item -ItemType Directory -Force -Path $downloadsRoot, (Join-Path $runtimeRoot 'ffmpeg') | Out-Null
        $archivePath = Join-Path $downloadsRoot ([IO.Path]::GetFileName($manifest.ffmpeg.url))
        if (-not (Test-Path -LiteralPath $archivePath)) { Invoke-WebRequest -Uri $manifest.ffmpeg.url -OutFile $archivePath }
        if ((Get-FileHash -LiteralPath $archivePath -Algorithm SHA256).Hash -ne $manifest.ffmpeg.sha256) { throw 'FFmpeg archive checksum mismatch.' }
        & tar -xf $archivePath -C (Join-Path $runtimeRoot 'ffmpeg')
        if ($LASTEXITCODE -ne 0) { throw 'FFmpeg extraction failed.' }
    }
    & (Join-Path $projectRoot $manifest.python_executable) -E -s -X utf8 (Join-Path $runtimeRoot 'verify.py')
    if ($LASTEXITCODE -ne 0) { throw 'Irodori runtime verification failed.' }
}
finally {
    foreach ($name in $previousEnvironment.Keys) {
        if ($null -eq $previousEnvironment[$name]) {
            Remove-Item -LiteralPath "Env:$name" -ErrorAction SilentlyContinue
        }
        else {
            [Environment]::SetEnvironmentVariable($name, $previousEnvironment[$name], 'Process')
        }
    }
}
