# ----------------------------------------
# Start-FaceInference.ps1
# Windows PowerShell launcher for FaceInferenceApp
# ----------------------------------------

# 1) Move to the directory containing this script
Set-Location -Path $PSScriptRoot

# 2) Activate venv (PowerShell version)
#    Adjust if your venv name is not "venv"
$venvActivate = ".\venv\Scripts\Activate.ps1"
if (Test-Path $venvActivate) {
    & $venvActivate
} else {
    Write-Host "Could not find venv activation script at $venvActivate"
    exit 1
}

# 4) Define target logical CPUs (your presumed P-cores)
#    Here: 0, 2, 4, 6, 8, 10 --> ONLY THE MAIN THREAD OF EACH CORE, NO HYPERTHREADING
$targetLogicalCPUs =  @(0,2,4,6,8,10)

# Build the affinity mask from the CPU indices
[int64]$mask = 0
foreach ($cpu in $targetLogicalCPUs) {
    $mask = $mask -bor (1 -shl $cpu)
}

Write-Host "➡ Starting FaceInferenceApp pinned to logical CPUs: $($targetLogicalCPUs -join ', ')"
Write-Host "   Affinity mask (decimal): $mask"
Write-Host "   Affinity mask (binary) : $([Convert]::ToString($mask, 2))"

# 5) Prepare process start info for Python + uvicorn
$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName = "python"  # uses the venv's python after activation
# DEV mode (http):
#$psi.Arguments = "-m uvicorn main:app --host 127.0.0.1 --port 8001 --workers 1 --no-access-log"
# PROD mode (https):
$psi.Arguments = "-m uvicorn main:app --host 127.0.0.1 --port 8001 --workers 1 --no-access-log --ssl-keyfile localhost-key.pem --ssl-certfile localhost.pem"
$psi.WorkingDirectory = (Get-Location).Path
$psi.UseShellExecute = $false      # inherit current env (venv, vars, etc.)
$psi.RedirectStandardOutput = $false
$psi.RedirectStandardError  = $false

# 6) Start process and then set affinity
$proc = [System.Diagnostics.Process]::Start($psi)
if ($null -eq $proc) {
    Write-Host "Failed to start FaceInferenceApp process."
    exit 1
}

# Set CPU affinity to the chosen cores
$proc.ProcessorAffinity = $mask

Write-Host "FaceInferenceApp started. PID: $($proc.Id)"
Write-Host "Affinity set to CPUs: $($targetLogicalCPUs -join ', ')"

# 7) (Optional) Wait here until the app exits
$proc.WaitForExit()
Write-Host "FaceInferenceApp has exited with code $($proc.ExitCode)"
