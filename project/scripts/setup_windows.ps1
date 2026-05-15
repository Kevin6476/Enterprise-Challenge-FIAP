$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")
Set-Location -LiteralPath $ProjectRoot

New-Item -ItemType Directory -Force -Path "reports" | Out-Null

if (-not (Test-Path -LiteralPath ".venv\Scripts\python.exe")) {
    Write-Host "Creating local virtual environment (.venv)..."
    python -m venv .venv
}

Write-Host "Upgrading pip..."
& ".\.venv\Scripts\python.exe" -m pip install --upgrade pip *> "reports\pip_upgrade.log"

Write-Host "Installing project dependencies..."
& ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt *> "reports\pip_install.log"

Write-Host "Verifying Python packages..."
@'
import pandas
import pyarrow
import sklearn
import xgboost

print("Environment OK")
print(f"pandas={pandas.__version__}")
print(f"pyarrow={pyarrow.__version__}")
print(f"scikit-learn={sklearn.__version__}")
print(f"xgboost={xgboost.__version__}")
'@ | & ".\.venv\Scripts\python.exe" - *> "reports\environment_check.txt"

Get-Content -LiteralPath "reports\environment_check.txt"
Write-Host "Setup completed."
