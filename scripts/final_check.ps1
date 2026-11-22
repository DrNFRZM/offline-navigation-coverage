<#
.SYNOPSIS
  Verify the finished research repository without running the training grid.

.DESCRIPTION
  Quick mode (default, about 5 minutes):
    1. integrity checks: essential files, Python/package versions against requirements-lock.txt,
       syntax and imports, dataset SHA-256 checksums, processed cache, checkpoint hashes,
       publication readiness (what git would include, ignored local material)
    2. `offnav verify` for the main study, pilot and smoke results (manifests, counts,
       per-episode -> summary recomputation, fingerprints)
    3. the automated test suite (unit + integration on the real data and simulator)
    4. independent recomputation of every main-study result from the per-episode records
       (including the post hoc failure-aware times)
    5. the post hoc failure-aware reanalysis rerun and compared byte for byte with results/main/reanalysis/
  -Full additionally (about 10 more minutes; GPU recommended):
    6. re-evaluates six saved policies, regenerates all tables in a temporary copy, reruns the
       complete smoke pipeline from scratch with the current code (all 20 runs must reproduce the
       saved records and pass the reuse gate) and trains one real-data job on the CPU.
  Nothing under results/ or artifacts/ is modified, except that `offnav verify` rewrites each
  results/<mode>/verification.json (its check report). Temporary files go to %TEMP%\offnav_final_check.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\final_check.ps1
.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\final_check.ps1 -Full
.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\final_check.ps1 -Python C:\path\to\python.exe
#>
param(
    [switch]$Full,
    [string]$Python = ""
)

$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
if (-not $Python) {
    $candidate = Join-Path $root ".venv\Scripts\python.exe"
    if (Test-Path $candidate) { $Python = $candidate } else { $Python = "python" }
}
$env:PYTHONDONTWRITEBYTECODE = "1"
$env:PYTHONWARNINGS = "ignore"
$env:PYTHONPATH = Join-Path $root "src"
$work = Join-Path $env:TEMP "offnav_final_check"
New-Item -ItemType Directory -Force $work | Out-Null
$summary = New-Object System.Collections.ArrayList

function Invoke-Check([string]$Name, [string[]]$Arguments) {
    Write-Host ""
    Write-Host "==== $Name" -ForegroundColor Cyan
    $start = Get-Date
    & $Python @Arguments
    $code = $LASTEXITCODE
    $secs = [math]::Round(((Get-Date) - $start).TotalSeconds, 0)
    if ($code -eq 0) { $status = "PASS" } else { $status = "FAIL" }
    [void]$summary.Add([pscustomobject]@{ Check = $Name; Result = $status; Seconds = $secs })
}

Write-Host "Repository: $root"
Write-Host "Python:     $Python"
& $Python --version

Invoke-Check "Files, environment, datasets, checkpoints, publication readiness" @("scripts/audit/integrity_checks.py", "--out", "$work\integrity.json")
Invoke-Check "Saved results: main study (offnav verify)" @("-m", "offnav.cli", "verify", "--config", "configs/main.json")
Invoke-Check "Saved results: pilot (offnav verify)" @("-m", "offnav.cli", "verify", "--config", "configs/pilot.json")
Invoke-Check "Saved results: smoke (offnav verify)" @("-m", "offnav.cli", "verify", "--config", "configs/smoke.json")
Invoke-Check "Automated tests (pytest)" @("-m", "pytest", "-q", "-p", "no:cacheprovider")
Invoke-Check "Independent recomputation of main-study results" @("scripts/audit/independent_recompute.py", "--out", "$work\evidence")
Invoke-Check "Post hoc failure-aware reanalysis reproduces its saved files" @("scripts/audit/reproduce_checks.py", "--reanalysis", "--tmp", "$work\repro", "--evidence", "$work\reanalysis_check.json")
if ($Full) {
    Invoke-Check "Reproduction: re-evaluation, table regeneration, smoke rerun, CPU job" @(
        "scripts/audit/reproduce_checks.py", "--reeval", "--tables", "--smoke", "--cpu-job",
        "--tmp", "$work\repro", "--evidence", "$work\reproduction.json")
}

Write-Host ""
Write-Host "==== Summary" -ForegroundColor Cyan
$summary | Format-Table -AutoSize | Out-String | Write-Host
$failed = @($summary | Where-Object { $_.Result -ne "PASS" }).Count
if ($failed -eq 0) {
    Write-Host "ALL CHECKS PASSED" -ForegroundColor Green
    exit 0
}
Write-Host "$failed check(s) FAILED - see the output above" -ForegroundColor Red
exit 1
