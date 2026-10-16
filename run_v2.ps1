# Kept so the old command still works: run.ps1 now handles every model version (v2, v3, ...).
powershell -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "run.ps1")
