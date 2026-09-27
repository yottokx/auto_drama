$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
Push-Location (Join-Path $projectRoot 'apps\web')
try {
    & npm.cmd run dev
    if ($LASTEXITCODE -ne 0) { throw 'Web dev server exited with an error.' }
}
finally { Pop-Location }
