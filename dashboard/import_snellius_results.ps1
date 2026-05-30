param(
    [Parameter(Mandatory = $true)]
    [string]$Archive,

    [string]$Python = ".\.venv\Scripts\python.exe"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $Archive)) {
    throw "Archive not found: $Archive"
}

tar -xzf $Archive

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python not found: $Python"
}

& $Python -m motndp.dashboard_data --output dashboard\data.js

Write-Host "Imported $Archive"
Write-Host "Regenerated dashboard\data.js"
