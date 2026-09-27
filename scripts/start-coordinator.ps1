[CmdletBinding()]
param([ValidateRange(1, 65535)][int]$Port = 8000, [string]$DataDir, [string]$TyranoDir)
$ErrorActionPreference = 'Stop'

function Assert-CoordinatorPortAvailable {
    param([int]$ListenPort)
    $probeSocket = [System.Net.Sockets.Socket]::new(
        [System.Net.Sockets.AddressFamily]::InterNetwork,
        [System.Net.Sockets.SocketType]::Stream,
        [System.Net.Sockets.ProtocolType]::Tcp
    )
    try {
        $probeSocket.ExclusiveAddressUse = $true
        $probeSocket.Bind([System.Net.IPEndPoint]::new([System.Net.IPAddress]::Loopback, $ListenPort))
    }
    catch [System.Net.Sockets.SocketException] {
        throw "127.0.0.1:$ListenPort を使用できません（ポートが使用中、または使用が制限されています）。制御サーバーが起動済みの場合は、その端末で Ctrl+C を押して終了してから、このスクリプトを再実行してください。別のアプリが使用している場合は -Port で空きポートを指定してください。プロセスは自動停止しません。"
    }
    finally {
        $probeSocket.Dispose()
    }
}

# Check before importing the app: even an unsuccessful uvicorn startup can migrate its database.
Assert-CoordinatorPortAvailable -ListenPort $Port
$projectRoot = Split-Path $PSScriptRoot -Parent
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw 'Run scripts/setup.ps1 -Components core first.' }
Push-Location $projectRoot
$previousDataDir = $env:AUTO_DRAMA_DATA_DIR
$previousTyranoDir = $env:AUTO_DRAMA_TYRANO_DIR
try {
    if ($DataDir) { $env:AUTO_DRAMA_DATA_DIR = $DataDir }
    if ($TyranoDir) { $env:AUTO_DRAMA_TYRANO_DIR = $TyranoDir }
    & $python -I -m uvicorn services.coordinator.app:app --app-dir $projectRoot --host 127.0.0.1 --port $Port
    if ($LASTEXITCODE -ne 0) { throw 'Coordinator exited with an error.' }
}
finally {
    $env:AUTO_DRAMA_DATA_DIR = $previousDataDir
    $env:AUTO_DRAMA_TYRANO_DIR = $previousTyranoDir
    Pop-Location
}

