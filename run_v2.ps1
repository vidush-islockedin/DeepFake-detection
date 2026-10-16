# Sets up and starts the Deepfake Detection app with the v2 model on Windows.
# Run from the DeepFake-detection folder:
#     powershell -ExecutionPolicy Bypass -File .\run_v2.ps1
# Safe to run again any time: it skips what is already done.

$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot
function Step($n, $text) { Write-Host "`n[$n/5] $text" -ForegroundColor Yellow }
function Fail($text) { Write-Host "`nSTOPPED: $text" -ForegroundColor Red; exit 1 }

Step 1 "Getting the latest code from GitHub"
git fetch origin
if ($LASTEXITCODE -ne 0) { Fail "git fetch failed. Check your internet connection." }
git reset --hard origin/main
if ($LASTEXITCODE -ne 0) { Fail "Could not update the code." }

Step 2 "Python environment and libraries (first time: 5-10 minutes)"
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) { Fail "Python not found. Install Python 3.10-3.12 and tick 'Add Python to PATH'." }
}
$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
& $py -m pip install --upgrade pip --quiet
& $py -m pip install -r requirements.txt --quiet
if ($LASTEXITCODE -ne 0) { Fail "Installing the libraries failed (see the message above)." }
& $py -m pip install facenet-pytorch==2.6.0 --no-deps --quiet

Step 3 "v2 model"
if (Test-Path "model\threshold.json") {
    Write-Host "v2 model is already in place."
} else {
    $zip = Get-ChildItem "$HOME\Downloads", "$HOME\Documents", "$HOME\Desktop" -Filter "model_v2*.zip" -Recurse -Depth 2 -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if (-not $zip) { Fail "model_v2.zip was not found in Downloads, Documents or Desktop. Put it in Downloads and run this again." }
    Write-Host "Using $($zip.FullName)"
    if (Test-Path "model") {
        $backup = "model_v1"
        if (Test-Path $backup) { $backup = "model_old_" + (Get-Date -Format "yyyyMMdd_HHmmss") }
        Rename-Item "model" $backup
        Write-Host "The old model folder was kept as $backup."
    }
    $tmp = Join-Path $env:TEMP "model_v2_unzip"
    if (Test-Path $tmp) { Remove-Item $tmp -Recurse -Force }
    Expand-Archive -Path $zip.FullName -DestinationPath $tmp -Force
    $weights = Get-ChildItem $tmp -Filter "model.safetensors" -Recurse | Select-Object -First 1
    if (-not $weights) { Fail "model.safetensors is not inside $($zip.Name)." }
    Move-Item $weights.Directory.FullName (Join-Path $PSScriptRoot "model")
    Remove-Item $tmp -Recurse -Force
}
foreach ($f in "config.json", "model.safetensors", "threshold.json") {
    if (-not (Test-Path "model\$f")) { Fail "model\$f is missing." }
}
Write-Host "Cut-off: $(Get-Content model\threshold.json)"

Step 4 "Checking the libraries load"
& $py -c "import torch, transformers, facenet_pytorch, fastapi; print('torch', torch.__version__, '| transformers', transformers.__version__)"
if ($LASTEXITCODE -ne 0) { Fail "A library did not load (see the message above)." }

Step 5 "Starting the app. The browser opens in about 30 seconds. Press Ctrl+C here to stop."
Start-Job { Start-Sleep 30; Start-Process "http://localhost:8000" } | Out-Null
& $py app.py
