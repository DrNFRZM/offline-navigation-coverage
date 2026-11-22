<#
.SYNOPSIS
  Reproduce the study end to end into a separate output folder (Windows PowerShell).

.DESCRIPTION
  Runs tests, then the smoke, pilot and main stages (prepare, train + evaluate grid, references,
  analysis, verification). Results and checkpoints are written under -OutDir, so the repository's
  saved results/ and artifacts/ stay untouched and can be compared with the new ones. Each stage is
  resumable: re-running the script continues where it stopped.
  Datasets are taken from data/ (or $env:OFFNAV_DATA); if the processed cache is missing,
  `offnav audit` downloads and audits them first (that step writes results/audit/ in the repository).
  Measured on the study machine (RTX 2060): smoke about 5 minutes, pilot about 30 minutes,
  main about 6 hours.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\reproduce.ps1 -OutDir C:\offnav_rerun
.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\reproduce.ps1 -OutDir C:\offnav_rerun -SkipPilot -SkipMain
#>
param(
    [string]$OutDir = (Join-Path $env:TEMP "offnav_reproduction"),
    [string]$Python = "",
    [switch]$SkipPilot,
    [switch]$SkipMain
)

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
if (-not $Python) {
    $candidate = Join-Path $root ".venv\Scripts\python.exe"
    if (Test-Path $candidate) { $Python = $candidate } else { $Python = "python" }
}
$env:OFFNAV_RESULTS = Join-Path $OutDir "results"
$env:OFFNAV_ARTIFACTS = Join-Path $OutDir "artifacts"
$env:PYTHONPATH = Join-Path $root "src"
Write-Host "Writing results to $env:OFFNAV_RESULTS and checkpoints to $env:OFFNAV_ARTIFACTS"

function Invoke-Stage([string[]]$Arguments) {
    Write-Host ">> python $($Arguments -join ' ')" -ForegroundColor Cyan
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Stage failed with exit code $LASTEXITCODE" -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

$dataRoot = if ($env:OFFNAV_DATA) { $env:OFFNAV_DATA } else { Join-Path $root "data" }
if (-not (Test-Path (Join-Path $dataRoot "processed\medium_flat.npz"))) {
    Invoke-Stage @("-m", "offnav.cli", "audit")
}
Invoke-Stage @("-m", "pytest", "-q", "-p", "no:cacheprovider")

$modes = @("smoke")
if (-not $SkipPilot) { $modes += "pilot" }
if (-not $SkipMain) { $modes += "main" }
foreach ($mode in $modes) {
    $cfg = "configs/$mode.json"
    Invoke-Stage @("-m", "offnav.cli", "prepare", "--config", $cfg)
    Invoke-Stage @("-m", "offnav.cli", "grid", "--config", $cfg, "--workers", "3")
    Invoke-Stage @("-m", "offnav.cli", "references", "--config", $cfg)
    Invoke-Stage @("-m", "offnav.cli", "analyze", "--config", $cfg)
    Invoke-Stage @("-m", "offnav.cli", "verify", "--config", $cfg)
}
Write-Host "Done. Compare $env:OFFNAV_RESULTS with the repository's results/ folder." -ForegroundColor Green
