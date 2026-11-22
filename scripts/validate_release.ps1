# Clean-environment validation of the public release archive (added 2026-11-22).
#
# Extracts release/offline-navigation-coverage.zip into a fresh folder, creates a new Python 3.10
# virtual environment from requirements-lock-cpu.txt (CPU build of torch), installs the package and
# runs, in the extracted copy only:
#   1. layout check: one top-level folder; .gitignore, .gitattributes, pyproject.toml, src/ at its root
#   2. git tree hash of the copy (what a first commit of exactly these files will record)
#   3. unit tests:          pytest -m "not integration"   (no datasets)
#   4. full test suite:     pytest                          (no datasets: data tests must skip, none fail)
#   5. integrity checks:    scripts/audit/integrity_checks.py --allow-missing-data
#   6. independent recomputation of the main results from the records in the copy
#   7. with -DataDir: full test suite and `offnav verify` for smoke, pilot and main on the real data
# The working folder, its results and its datasets are only read.
#
# Usage (repository root, after `python scripts/export_public.py`):
#   powershell -ExecutionPolicy Bypass -File scripts\validate_release.ps1 [-WorkDir C:\tmp\offnav_release] [-DataDir data]

param(
    [string]$Archive = "release\offline-navigation-coverage.zip",
    [string]$WorkDir = (Join-Path $env:TEMP "offnav_release_check"),
    [string]$DataDir = "",
    [string]$Python = "py -3.10"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$ArchivePath = Resolve-Path $Archive
$results = New-Object System.Collections.Generic.List[object]

function Step($name, [scriptblock]$block) {
    Write-Host "`n=== $name" -ForegroundColor Cyan
    $t0 = Get-Date
    & $block
    $ok = ($LASTEXITCODE -eq 0) -or ($null -eq $LASTEXITCODE)
    $results.Add([pscustomobject]@{ step = $name; status = $(if ($ok) { "PASS" } else { "FAIL" }); seconds = [int]((Get-Date) - $t0).TotalSeconds })
    $global:LASTEXITCODE = 0
}

if (Test-Path $WorkDir) { Remove-Item -Recurse -Force $WorkDir }
New-Item -ItemType Directory -Force $WorkDir | Out-Null
Expand-Archive -Path $ArchivePath -DestinationPath (Join-Path $WorkDir "src") -Force
$top = @(Get-ChildItem -Force (Join-Path $WorkDir "src"))
$Repo = Join-Path (Join-Path $WorkDir "src") "offline-navigation-coverage"
$Venv = Join-Path $WorkDir "venv"
$EmptyData = Join-Path $WorkDir "nodata"
New-Item -ItemType Directory -Force $EmptyData | Out-Null
$Py = Join-Path $Venv "Scripts\python.exe"

Step "layout: single top-level folder with hidden files" {
    $need = ".gitignore", ".gitattributes", "pyproject.toml", "README.md", "LICENSE", "requirements-lock-cpu.txt", "src\offnav\cli.py", "PUBLIC_RELEASE.md"
    $missing = $need | Where-Object { -not (Test-Path (Join-Path $Repo $_)) }
    if ($top.Count -ne 1 -or $missing) { Write-Host "top-level entries: $($top.Name -join ', '); missing: $($missing -join ', ')"; $global:LASTEXITCODE = 1 }
    else { Write-Host "OK: offline-navigation-coverage/ with $((Get-ChildItem -Recurse -File -Force $Repo).Count) files" }
}

Step "git tree hash of the release copy (no commit is made)" {
    $env:GIT_DIR = Join-Path $WorkDir "tree.git"
    $env:GIT_WORK_TREE = $Repo
    git init -q
    # independent of the user's line-ending settings: .gitattributes alone decides normalisation
    git -c core.autocrlf=false -c core.safecrlf=false add -A
    $tree = git write-tree
    Remove-Item Env:GIT_DIR; Remove-Item Env:GIT_WORK_TREE
    Write-Host "tree $tree"
    Set-Content -Path (Join-Path $WorkDir "tree_hash.txt") -Value $tree -Encoding ascii
}

Step "fresh virtual environment from requirements-lock-cpu.txt" {
    Invoke-Expression "$Python -m venv `"$Venv`""
    # Python 3.10.2 ships pip 21.2, which cannot install a pyproject-only package in editable mode
    & $Py -m pip install --disable-pip-version-check -q pip==26.3
    & $Py -m pip install --disable-pip-version-check -q -r (Join-Path $Repo "requirements-lock-cpu.txt") --extra-index-url https://download.pytorch.org/whl/cpu
    if ($LASTEXITCODE -eq 0) { & $Py -m pip install --disable-pip-version-check -q -e $Repo --no-deps }
    if ($LASTEXITCODE -eq 0) { & $Py -c "import offnav, torch, minari, gymnasium, gymnasium_robotics, mujoco; print('imports OK; torch', torch.__version__, '; offnav from', offnav.__file__)" }
}

$env:PYTHONDONTWRITEBYTECODE = "1"
$env:PYTHONWARNINGS = "ignore"
Push-Location $Repo
try {
    $env:OFFNAV_DATA = $EmptyData
    Step "unit tests without datasets: pytest -m 'not integration'" { & $Py -m pytest -m "not integration" -q -p no:cacheprovider }
    Step "full test suite without datasets (data tests skip)" { & $Py -m pytest -q -p no:cacheprovider -rs }
    Step "integrity checks (--allow-missing-data)" { & $Py scripts\audit\integrity_checks.py --allow-missing-data }
    Step "independent recomputation from the records in the copy" { & $Py scripts\audit\independent_recompute.py --out (Join-Path $WorkDir "evidence") }
    if ($DataDir) {
        $env:OFFNAV_DATA = (Resolve-Path (Join-Path $Root $DataDir)).Path
        Step "full test suite with datasets" { & $Py -m pytest -q -p no:cacheprovider -rs }
        foreach ($mode in "smoke", "pilot", "main") {
            Step "offnav verify --config configs/$mode.json (with datasets)" { & $Py -m offnav.cli verify --config "configs/$mode.json" }
        }
        Step "offnav status --config configs/main.json" { & $Py -m offnav.cli status --config configs/main.json }
    }
} finally {
    Pop-Location
    Remove-Item Env:OFFNAV_DATA -ErrorAction SilentlyContinue
}

Write-Host "`n=== summary"
$results | Format-Table -AutoSize | Out-String | Write-Host
$results | ConvertTo-Json | Set-Content -Path (Join-Path $WorkDir "validate_release_summary.json") -Encoding utf8
$hash = (Get-FileHash -Algorithm SHA256 $ArchivePath).Hash.ToLower()
$treeFile = Join-Path $WorkDir "tree_hash.txt"
$record = [pscustomobject]@{
    archive = (Split-Path -Leaf $ArchivePath); archive_sha256 = $hash
    git_tree = $(if (Test-Path $treeFile) { (Get-Content $treeFile -Raw).Trim() } else { $null })
    validated_utc = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ"); with_datasets = [bool]$DataDir
    steps = $results; all_passed = -not ($results | Where-Object status -eq "FAIL")
}
# kept next to the archive (release/ is git-ignored), not inside the copy, whose hash it records
$record | ConvertTo-Json -Depth 4 | Set-Content -Path (Join-Path (Split-Path -Parent $ArchivePath) "validation_record.json") -Encoding utf8
Write-Host "archive sha256 $hash"
Write-Host "work folder    $WorkDir"
if ($results | Where-Object status -eq "FAIL") { exit 1 } else { exit 0 }
