<#
Download and prepare the Omniglot dataset for this repository.

What it does:
- Downloads the Omniglot GitHub archive (master.zip)
- Extracts it to a temporary directory
- Copies the contents of `python/images_background` and
  `python/images_evaluation` into `omniglot/images_all` inside the repo
- Leaves existing files in place and is safe to re-run

Usage (from repo root):
  powershell -ExecutionPolicy Bypass -File .\scripts\setup_omniglot.ps1

If you want to place the dataset in a different target folder, pass a path:
  powershell -ExecutionPolicy Bypass -File .\scripts\setup_omniglot.ps1 -TargetDir 'D:\datasets\omniglot'
#>

param(
  [string]$TargetDir = ''
)

try {
  $scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
} catch {
  $scriptDir = Get-Location
}

if ([string]::IsNullOrEmpty($TargetDir)) {
  $repoRoot = Resolve-Path -Path $scriptDir | Select-Object -ExpandProperty Path
  # repo root is the script directory's parent (script lives in scripts/)
  $repoRoot = Split-Path -Parent $repoRoot
  $target = Join-Path $repoRoot 'omniglot'
} else {
  $target = Resolve-Path -Path $TargetDir | Select-Object -ExpandProperty Path
}

Write-Host "Target Omniglot folder: $target"

$zipUrl = 'https://github.com/brendenlake/omniglot/archive/refs/heads/master.zip'
$zipLocal = Join-Path $env:TEMP 'omniglot-master.zip'
$extractDir = Join-Path $env:TEMP 'omniglot_master'

function Ensure-Dir($p) {
  if (-not (Test-Path $p)) {
    New-Item -ItemType Directory -Path $p -Force | Out-Null
  }
}

Ensure-Dir $target
Ensure-Dir (Join-Path $target 'images_all')

Write-Host "Downloading Omniglot archive..."
if (Test-Path $zipLocal) { Remove-Item $zipLocal -Force }
Invoke-WebRequest -Uri $zipUrl -OutFile $zipLocal -UseBasicParsing

Write-Host "Extracting archive..."
if (Test-Path $extractDir) { Remove-Item $extractDir -Recurse -Force -ErrorAction SilentlyContinue }
Expand-Archive -Path $zipLocal -DestinationPath $extractDir -Force

Write-Host "Searching extracted archive for Omniglot image folders..."
# Search for images_background / images_evaluation anywhere under the extracted tree
$archiveRoot = Join-Path $extractDir 'omniglot-master'
if (-not (Test-Path $archiveRoot)) { $archiveRoot = $extractDir }

$foundSrc1 = Get-ChildItem -Path $archiveRoot -Recurse -Directory -ErrorAction SilentlyContinue | Where-Object { $_.Name -ieq 'images_background' } | Select-Object -First 1
$foundSrc2 = Get-ChildItem -Path $archiveRoot -Recurse -Directory -ErrorAction SilentlyContinue | Where-Object { $_.Name -ieq 'images_evaluation' } | Select-Object -First 1

if ($null -eq $foundSrc1 -and $null -eq $foundSrc2) {
  Write-Error "Could not find images_background or images_evaluation in the extracted archive. Inspect $archiveRoot"
  exit 1
}

if ($foundSrc1) {
  $src1 = $foundSrc1.FullName
  Write-Host "Found images_background at: $src1"
  Get-ChildItem -Path $src1 -Directory | ForEach-Object {
    $dest = Join-Path (Join-Path $target 'images_all') $_.Name
    Write-Host "Copying folder: $($_.FullName) -> $dest"
    Copy-Item -Path $_.FullName -Destination $dest -Recurse -Force
  }
} else { Write-Warning "images_background not found in archive" }

if ($foundSrc2) {
  $src2 = $foundSrc2.FullName
  Write-Host "Found images_evaluation at: $src2"
  Get-ChildItem -Path $src2 -Directory | ForEach-Object {
    $dest = Join-Path (Join-Path $target 'images_all') $_.Name
    Write-Host "Copying folder: $($_.FullName) -> $dest"
    Copy-Item -Path $_.FullName -Destination $dest -Recurse -Force
  }
} else { Write-Warning "images_evaluation not found in archive" }

Write-Host "Cleaning up temporary files..."
Remove-Item $zipLocal -Force -ErrorAction SilentlyContinue
Remove-Item $extractDir -Recurse -Force -ErrorAction SilentlyContinue

Write-Host "Omniglot prepared at: $($target)\images_all"
Write-Host "Done. You can now run run_eval.py with --dataset omniglot"
