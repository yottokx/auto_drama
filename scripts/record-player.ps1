param(
    [string]$ProjectId,
    [string]$Url,
    [string]$BaseUrl = 'http://127.0.0.1:8000',
    [string]$OutputDir,
    [ValidateSet('msedge', 'chrome', 'chromium')][string]$Browser = 'msedge',
    [int]$Fps = 15,
    [double]$MaxSeconds = 7200,
    [switch]$SingleChapter,
    [switch]$Foreground,
    # 録画する画面の設定。鑑賞画面の「設定」と同じ項目です。
    [ValidateSet('small', 'standard', 'large')][string]$TextSize = 'standard',
    [ValidateSet('gothic', 'mincho')][string]$Typeface = 'gothic',
    [ValidateRange(0, 100)][int]$TextSpeed = 62,
    [ValidateRange(0.3, 4.0)][double]$AutoWait = 1.4,
    [ValidateRange(0, 100)][int]$VoiceVolume = 100,
    [ValidateRange(0, 100)][int]$BgmVolume = 100,
    [ValidateRange(30, 100)][int]$WindowOpacity = 80
)
$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
$python = Join-Path $root '.venv/Scripts/python.exe'
if ([bool]$ProjectId -eq [bool]$Url) { throw 'ProjectId または Url のどちらか一方を指定してください。' }
if (-not $OutputDir) {
    $OutputDir = Join-Path $root ('outputs/recording-' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff'))
}
$OutputDir = [System.IO.Path]::GetFullPath($OutputDir)
if (Test-Path -LiteralPath $OutputDir) { throw "保存先は既に存在します: $OutputDir" }
$parent = Split-Path $OutputDir -Parent
New-Item -ItemType Directory -Path $parent -Force | Out-Null
$arguments = @('-u', '-X', 'utf8', (Join-Path $PSScriptRoot 'record/record_player.py'),
    '--output-dir', $OutputDir, '--base-url', $BaseUrl, '--browser', $Browser,
    '--fps', "$Fps", '--max-seconds', $MaxSeconds.ToString([cultureinfo]::InvariantCulture),
    '--text-size', $TextSize, '--typeface', $Typeface, '--text-speed', "$TextSpeed",
    '--auto-wait', $AutoWait.ToString([cultureinfo]::InvariantCulture),
    '--voice-volume', "$VoiceVolume", '--bgm-volume', "$BgmVolume", '--window-opacity', "$WindowOpacity")
if ($ProjectId) { $arguments += @('--project-id', $ProjectId) }
if ($Url) { $arguments += @('--url', $Url) }
if ($SingleChapter) { $arguments += '--single-chapter' }
if ($Foreground) {
    & $python @arguments
    exit $LASTEXITCODE
}
# Start-Process joins ArgumentList into one Windows command line. Quote every argument.
$quoted = $arguments | ForEach-Object {
    if ($_ -match '"') { throw '引数に二重引用符は指定できません。' }
    '"' + ($_ -replace '(\\+)$', '$1$1') + '"'
}
$process = Start-Process -FilePath $python -ArgumentList ($quoted -join ' ') -WorkingDirectory $root `
    -WindowStyle Hidden -PassThru -RedirectStandardOutput ($OutputDir + '.stdout.log') `
    -RedirectStandardError ($OutputDir + '.stderr.log')
Write-Output "録画プロセスを起動しました。PID: $($process.Id)"
Write-Output "保存先: $OutputDir"
Write-Output "状態: $OutputDir/status.json"
Write-Output "エラーログ: $OutputDir.stderr.log"
Write-Output "停止: New-Item -ItemType File -Path '$OutputDir/stop.request'"
