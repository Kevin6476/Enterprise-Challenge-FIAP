$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")
Set-Location -LiteralPath $ProjectRoot

if (-not (Test-Path -LiteralPath ".venv\Scripts\python.exe")) {
    throw "Virtual environment not found. Run: powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1"
}

New-Item -ItemType Directory -Force -Path "reports" | Out-Null

Write-Host "Running full ML/data pipeline..."
$PreviousErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
& ".\.venv\Scripts\python.exe" -m src.main *> "reports\pipeline_run.log"
$PipelineExitCode = $LASTEXITCODE
$ErrorActionPreference = $PreviousErrorActionPreference
if ($PipelineExitCode -ne 0) {
    Get-Content -LiteralPath "reports\pipeline_run.log" | Select-Object -Last 80
    exit $PipelineExitCode
}

Write-Host "Building dashboard and validating MVP..."
$PreviousErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
& ".\.venv\Scripts\python.exe" "scripts\build_and_validate.py" *> "reports\demo_validation.log"
$ValidationExitCode = $LASTEXITCODE
$ErrorActionPreference = $PreviousErrorActionPreference
if ($ValidationExitCode -ne 0) {
    Get-Content -LiteralPath "reports\demo_validation.log" | Select-Object -Last 120
    exit $ValidationExitCode
}

Get-Content -LiteralPath "reports\demo_validation.log" | Select-Object -Last 80

$DashboardPath = Join-Path $ProjectRoot "app\index.html"
Write-Host ""
Write-Host "Demo ready."
Write-Host "Open this file in your browser:"
Write-Host $DashboardPath
