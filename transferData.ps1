# Save as: Copy-Recordings-To-X9PRO.ps1
$ErrorActionPreference = "Stop"

# -----------------------------
# CONFIG
# -----------------------------
$Source      = Join-Path $env:USERPROFILE "Documents\GitHub\FaceInferenceApp\ProcessedRecordings"
$VolumeLabel = "X9PRO"

# Verification modes: "none" | "payload" | "all"
$VerifyMode  = "payload"

# Payload patterns (used when VerifyMode = "payload")
$PayloadPatterns = @("*.mraw.zst", "*_infer.msgpk")

# -----------------------------
# PRECHECKS
# -----------------------------
if (-not (Test-Path -LiteralPath $Source)) {
    Write-Host "SOURCE NOT FOUND:" -ForegroundColor Red
    Write-Host "  $Source"
    Read-Host "Press Enter to exit" | Out-Null
    exit 1
}

# -----------------------------
# FIND EXTERNAL DRIVE BY LABEL
# -----------------------------
$vol = Get-Volume | Where-Object { $_.FileSystemLabel -eq $VolumeLabel } | Select-Object -First 1
if (-not $vol -or -not $vol.DriveLetter) {
    Write-Host "EXTERNAL DRIVE '$VolumeLabel' NOT FOUND." -ForegroundColor Red
    Read-Host "Press Enter to exit" | Out-Null
    exit 1
}

$DriveRoot = "$($vol.DriveLetter):\"
$Dest      = Join-Path $DriveRoot "Recordings"
$LogDir    = Join-Path $DriveRoot "TransferLogs"

New-Item -ItemType Directory -Force -Path $Dest   | Out-Null
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$stamp   = Get-Date -Format "yyyyMMdd_HHmmss"
$Log     = Join-Path $LogDir "robocopy_Recordings_$stamp.log"
$HashLog = Join-Path $LogDir "verify_hashes_$stamp.log"

Write-Host ""
Write-Host "Copying recordings to $VolumeLabel ($DriveRoot)" -ForegroundColor Cyan
Write-Host "FROM: $Source"
Write-Host "TO  : $Dest"
Write-Host ""

# -----------------------------
# ROBOCOPY
# -----------------------------
robocopy `
  "$Source" `
  "$Dest" `
  /E `
  /Z `
  /R:3 `
  /W:5 `
  /FFT `
  /DCOPY:T `
  /COPY:DAT `
  /NP `
  /XJ `
  /MT:1 `
  /TEE `
  /LOG:"$Log"

$code = $LASTEXITCODE
Write-Host "Robocopy exit code: $code" -ForegroundColor Yellow

if ($code -ge 8) {
    Write-Host "TRANSFER FAILED. DO NOT UPLOAD." -ForegroundColor Red
    Write-Host "See log: $Log"
    Read-Host "Press Enter to exit" | Out-Null
    exit 1
}

# Give USB cache time to flush
Start-Sleep -Seconds 3

# -----------------------------
# VERIFICATION
# -----------------------------
if ($VerifyMode -ne "none") {

    Write-Host ""
    Write-Host "Verification mode: $VerifyMode (SHA-256)" -ForegroundColor Cyan
    Write-Host "Hash log: $HashLog"
    Write-Host ""

    if ($VerifyMode -eq "all") {
        $srcFiles = Get-ChildItem -LiteralPath $Source -File -Recurse -Force
    }
    elseif ($VerifyMode -eq "payload") {

        $srcFiles = @()

        foreach ($pat in $PayloadPatterns) {
            $srcFiles += Get-ChildItem -LiteralPath $Source -File -Recurse -Force -Filter $pat
        }

        $srcFiles = $srcFiles | Sort-Object FullName -Unique
    }
    else {
        Write-Host "Unknown VerifyMode '$VerifyMode'" -ForegroundColor Red
        exit 1
    }

    if ($srcFiles.Count -eq 0) {
        Write-Host "No files to verify." -ForegroundColor Yellow
        exit 0
    }

    "VERIFY START: $(Get-Date -Format o)" | Out-File $HashLog -Encoding UTF8
    "MODE: $VerifyMode"                  | Out-File $HashLog -Append
    "SOURCE: $Source"                    | Out-File $HashLog -Append
    "DEST  : $Dest"                      | Out-File $HashLog -Append
    ""                                   | Out-File $HashLog -Append

    $missing = @()
    $bad     = @()

    $i = 0
    foreach ($sf in $srcFiles) {
        $i++
        $rel = $sf.FullName.Substring($Source.Length).TrimStart("\")
        $df  = Join-Path $Dest $rel

        Write-Host ("Hashing {0}/{1}: {2}" -f $i, $srcFiles.Count, $rel)

        if (-not (Test-Path -LiteralPath $df)) {
            $missing += $rel
            "MISSING_DST: $rel" | Out-File $HashLog -Append
            continue
        }

        $h1 = (Get-FileHash -LiteralPath $sf.FullName -Algorithm SHA256).Hash
        $h2 = (Get-FileHash -LiteralPath $df           -Algorithm SHA256).Hash

        if ($h1 -ne $h2) {
            $bad += $rel
            "MISMATCH: $rel" | Out-File $HashLog -Append
        }
        else {
            "OK: $rel $h1" | Out-File $HashLog -Append
        }
    }

    "" | Out-File $HashLog -Append
    "VERIFY END: $(Get-Date -Format o)" | Out-File $HashLog -Append

    if ($missing.Count -gt 0 -or $bad.Count -gt 0) {
        Write-Host "VERIFICATION FAILED." -ForegroundColor Red
        Write-Host "Missing files   : $($missing.Count)"
        Write-Host "Hash mismatches : $($bad.Count)"
        Write-Host "See: $HashLog"
        Read-Host "Press Enter to exit" | Out-Null
        exit 1
    }

    Write-Host "VERIFICATION OK (SHA-256 matched)." -ForegroundColor Green
}

Write-Host ""
Write-Host "TRANSFER OK. SAFE TO UPLOAD." -ForegroundColor Green
Read-Host "Press Enter to close" | Out-Null