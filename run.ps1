# Sets up and starts the Deepfake Detection app on Windows.
# Run from the DeepFake-detection folder:
#     powershell -ExecutionPolicy Bypass -File .\run.ps1            newest model (finds model_v3.zip in Downloads etc.)
#     powershell -ExecutionPolicy Bypass -File .\run.ps1 -Use 2     go back to v2 (and -Use 3 to return to v3)
# A model that is replaced is kept as model_v<N>, so you can always switch back. Safe to run again any time.
param([int]$Use = 0)

$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot
function Step($n, $text) { Write-Host "`n[$n/5] $text" -ForegroundColor Yellow }
function Fail($text) { Write-Host "`nSTOPPED: $text" -ForegroundColor Red; exit 1 }
function CurrentVersion {
    if (Test-Path "model\version.txt") { return [int]((Get-Content "model\version.txt" -Raw).Trim().TrimStart("v")) }
    if (Test-Path "model\threshold.json") { return 2 }          # v2 had a cut-off file but no version file
    if (Test-Path "model") { return 1 }
    return 0
}

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

Step 3 "Model"
function FindZip($version) {
    Get-ChildItem "$HOME\Downloads", "$HOME\Documents", "$HOME\Desktop" -Filter "model_v*.zip" -Recurse -Depth 2 -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match '^model_v(\d+)' } |
        ForEach-Object { [pscustomobject]@{ File = $_; V = [int]([regex]::Match($_.Name, '^model_v(\d+)').Groups[1].Value) } } |
        Where-Object { $version -eq 0 -or $_.V -eq $version } |
        Sort-Object -Property V, @{ Expression = { $_.File.LastWriteTime } } -Descending | Select-Object -First 1
}
function KeepCurrent {
    if (Test-Path "model") {
        $backup = "model_v$(CurrentVersion)"
        if (Test-Path $backup) { $backup = "model_old_" + (Get-Date -Format "yyyyMMdd_HHmmss") }
        Rename-Item "model" $backup
        Write-Host "The model in use was kept as $backup."
    }
}
function InstallZip($zip) {
    Write-Host "Installing model v$($zip.V) from $($zip.File.FullName)"
    KeepCurrent
    $tmp = Join-Path $env:TEMP "model_unzip"
    if (Test-Path $tmp) { Remove-Item $tmp -Recurse -Force }
    Expand-Archive -Path $zip.File.FullName -DestinationPath $tmp -Force
    $weights = Get-ChildItem $tmp -Filter "model.safetensors" -Recurse | Select-Object -First 1
    if (-not $weights) { Fail "model.safetensors is not inside $($zip.File.Name)." }
    Move-Item $weights.Directory.FullName (Join-Path $PSScriptRoot "model")
    Remove-Item $tmp -Recurse -Force
    if (-not (Test-Path "model\version.txt")) { Set-Content "model\version.txt" "v$($zip.V)" }
}
$cur = CurrentVersion
if ($Use -gt 0) {                                   # a chosen version: a kept folder first, else its zip
    if ($Use -eq $cur) {
        Write-Host "Model v$Use is already in use."
    } elseif (Test-Path "model_v$Use\model.safetensors") {
        KeepCurrent
        Rename-Item "model_v$Use" "model"
        if (-not (Test-Path "model\version.txt")) { Set-Content "model\version.txt" "v$Use" }
        Write-Host "Switched to model v$Use."
    } else {
        $zip = FindZip $Use
        if (-not $zip) { Fail "No model_v$Use folder here and no model_v$Use.zip in Downloads, Documents or Desktop." }
        InstallZip $zip
    }
} else {                                            # newest: install a downloaded zip if it is newer
    $zip = FindZip 0
    if ($zip -and $zip.V -gt $cur) {
        InstallZip $zip
    } elseif ($cur -eq 0) {
        Fail "No model found. Put model_v3.zip (from Kaggle) in Downloads and run this again."
    } else {
        Write-Host "Using model v$cur (no newer model_v*.zip found)."
    }
}
foreach ($f in "config.json", "model.safetensors", "threshold.json") {
    if (-not (Test-Path "model\$f")) { Fail "model\$f is missing." }
}
Write-Host "Model v$(CurrentVersion), cut-off: $(Get-Content model\threshold.json)"

Step 4 "Checking the libraries load"
& $py -c "import torch, transformers, facenet_pytorch, fastapi; print('torch', torch.__version__, '| transformers', transformers.__version__)"
if ($LASTEXITCODE -ne 0) { Fail "A library did not load (see the message above)." }

Step 5 "Starting the app. The browser opens in about 30 seconds. Press Ctrl+C here to stop."
Start-Job { Start-Sleep 30; Start-Process "http://localhost:8000" } | Out-Null
& $py app.py
