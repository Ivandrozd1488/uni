<#
.SYNOPSIS
  unified_ml — Windows Installer (PowerShell)

.DESCRIPTION
  Downloads a prebuilt binary package from GitHub Releases (SHA256 verified),
  installs it to a versioned directory, and optionally adds it to the user
  CMAKE_PREFIX_PATH.  Falls back to source build if download fails and CMake
  is available.

.PARAMETER Prefix
  Installation root. Default: C:\unified_ml

.PARAMETER Version
  Library version to install. Default: latest

.PARAMETER Repo
  GitHub owner/repo. Default: your-org/unified_ml

.PARAMETER Mode
  auto | download | source. Default: auto

.PARAMETER NoVersionedDir
  Install flat into Prefix instead of Prefix\VERSION

.PARAMETER NoSymlink
  Skip creating Prefix\current junction/symlink

.PARAMETER AddToPath
  Add install dir to user CMAKE_PREFIX_PATH permanently

.PARAMETER DryRun
  Print actions without executing them

.EXAMPLE
  .\install.ps1
  .\install.ps1 -Prefix "D:\libs" -Version "1.2.0"
  .\install.ps1 -Mode source -Prefix "C:\unified_ml"
  .\install.ps1 -AddToPath -DryRun
#>
[CmdletBinding()]
param(
  [string] $Prefix   = "C:\unified_ml",
  [string] $Version    = "latest",
  [string] $Repo     = "your-org/unified_ml",
  [ValidateSet("auto","download","source")]
  [string] $Mode     = "auto",
  [switch] $NoVersionedDir,
  [switch] $NoSymlink,
  [switch] $AddToPath,
  [switch] $DryRun
)

$ErrorActionPreference = "Stop"
$ProgressPreference  = "SilentlyContinue" # faster Invoke-WebRequest

# Colours             
function Write-Step  { Write-Host ""; Write-Host "━━ $args" -ForegroundColor Cyan }
function Write-Ok  { Write-Host "[✓] $args"  -ForegroundColor Green }
function Write-Info  { Write-Host "[→] $args"  -ForegroundColor White }
function Write-Warn  { Write-Host "[!] $args"  -ForegroundColor Yellow }
function Write-Fail  { Write-Host "[✗] $args"  -ForegroundColor Red; exit 1 }
function Write-Dry { Write-Host "  [dry-run] $args" -ForegroundColor DarkGray }

# Banner             
Write-Host ""
Write-Host "  unified_ml Installer for Windows" -ForegroundColor White
Write-Host "  Prefix  : $Prefix"
Write-Host "  Mode  : $Mode"
Write-Host "  Version : $Version"
if ($DryRun) { Write-Host "  DRY RUN — no changes will be made" -ForegroundColor Yellow }

# Resolve version from GitHub API        
function Resolve-Version {
  if ($Version -ne "latest") { return $Version }
  Write-Step "Resolving latest version"
  try {
    $api  = "https://api.github.com/repos/$Repo/releases/latest"
    $resp = Invoke-RestMethod -Uri $api -UseBasicParsing -ErrorAction Stop
    $ver  = $resp.tag_name -replace '^v',''
    Write-Ok "Latest: $ver"
    return $ver
  } catch {
    Write-Warn "GitHub API error: $_"
    return $null
  }
}

# SHA256 verification           
function Confirm-Sha256 {
  param([string]$FilePath, [string]$Expected)
  if ([string]::IsNullOrWhiteSpace($Expected)) {
    Write-Warn "No expected hash — skipping integrity check"
    return
  }
  $actual = (Get-FileHash $FilePath -Algorithm SHA256).Hash.ToLower()
  if ($Expected.ToLower() -ne $actual) {
    Write-Fail "SHA256 mismatch!`n  Expected: $Expected`n  Actual  : $actual`n  File  : $FilePath"
  }
  Write-Ok "SHA256 verified: $actual"
}

# Effective install prefix          
function Get-EffectivePrefix([string]$ver) {
  if ($NoVersionedDir) { return $Prefix }
  return Join-Path $Prefix $ver
}

# Prebuilt download          
function Install-Prebuilt {
  $ver = Resolve-Version
  if (-not $ver) { return $false }

  $archive  = "unified_ml-$ver-windows-x64.zip"
  $baseUrl  = "https://github.com/$Repo/releases/download/v$ver"
  $url  = "$baseUrl/$archive"
  $shaUrl = "$url.sha256"
  $tmp  = Join-Path $env:TEMP "unified_ml_$ver_$(Get-Random)"
  $dest   = Get-EffectivePrefix $ver

  Write-Step "Downloading prebuilt binary"
  Write-Info "URL  : $url"
  Write-Info "Dest : $dest"

  if ($DryRun) {
    Write-Dry "Invoke-WebRequest $url → $tmp\$archive"
    Write-Dry "Verify SHA256"
    Write-Dry "Expand-Archive → $dest"
    $script:ResolvedVersion = $ver
    return $true
  }

  New-Item -ItemType Directory -Path $tmp -Force | Out-Null

  try {
    Invoke-WebRequest -Uri $url -OutFile "$tmp\$archive" -UseBasicParsing
  } catch {
    Write-Warn "Download failed: $_"
    Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
    return $false
  }
  Write-Ok "Downloaded $('{0:N1} MB' -f ((Get-Item "$tmp\$archive").Length / 1MB))"

  # SHA256
  try {
    $shaContent = (Invoke-WebRequest -Uri $shaUrl -UseBasicParsing).Content.Trim()
    $expectedSha = ($shaContent -split '\s+')[0]
    Confirm-Sha256 -FilePath "$tmp\$archive" -Expected $expectedSha
  } catch {
    Write-Warn "Could not fetch SHA256 ($shaUrl): $_"
  }

  # Extract
  Write-Step "Extracting"
  Expand-Archive -Path "$tmp\$archive" -DestinationPath $tmp -Force
  $extracted = Join-Path $tmp "unified_ml-$ver-windows-x64"
  if (-not (Test-Path $extracted)) {
    Write-Warn "Expected directory not found: $extracted"
    return $false
  }

  # Install
  Write-Step "Installing to $dest"
  New-Item -ItemType Directory -Path $dest -Force | Out-Null
  Copy-Item -Path "$extracted\*" -Destination $dest -Recurse -Force
  Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
  Write-Ok "Installed ✓"

  $script:ResolvedVersion = $ver
  return $true
}

# Source build            
function Build-FromSource([string]$ver) {
  Write-Step "Building from source"

  if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) {
    Write-Fail "cmake not found. Download: https://cmake.org/download/"
  }
  $cmakeVer = (cmake --version | Select-Object -First 1) -replace '.*(\d+\.\d+\.\d+).*','$1'
  Write-Ok "CMake $cmakeVer"

  $scriptDir = $PSScriptRoot
  if (-not $scriptDir) { $scriptDir = $PWD }
  $buildDir  = Join-Path $env:TEMP "unified_ml_build_$(Get-Random)"
  $dest  = Get-EffectivePrefix $ver

  if ($DryRun) {
    Write-Dry "cmake -S $scriptDir -B $buildDir -DCMAKE_INSTALL_PREFIX=$dest ..."
    Write-Dry "cmake --build $buildDir --config Release"
    Write-Dry "cmake --install $buildDir"
    return
  }

  Write-Info "Configure..."
  cmake -S $scriptDir -B $buildDir `
    -DCMAKE_BUILD_TYPE=Release `
    -DCMAKE_INSTALL_PREFIX="$dest" `
    -DUNIFIED_ML_BUILD_TESTS=OFF `
    -DUNIFIED_ML_BUILD_BENCHMARKS=OFF `
    -DUNIFIED_ML_BUILD_SHARED=ON `
    -DUNIFIED_ML_BUILD_STATIC=ON `
    -DUNIFIED_ML_INSTALL=ON `
    -DUNIFIED_ML_ENABLE_OPENMP=ON `
    -DUNIFIED_ML_ENABLE_AVX=ON `
    -DUNIFIED_ML_VERSIONED_INSTALL=OFF
  if ($LASTEXITCODE -ne 0) { Write-Fail "Configure failed (exit $LASTEXITCODE)" }

  Write-Info "Compiling..."
  cmake --build $buildDir --config Release --parallel
  if ($LASTEXITCODE -ne 0) { Write-Fail "Build failed (exit $LASTEXITCODE)" }

  Write-Info "Installing to $dest ..."
  cmake --install $buildDir --config Release
  if ($LASTEXITCODE -ne 0) { Write-Fail "Install failed (exit $LASTEXITCODE)" }

  Remove-Item -Recurse -Force $buildDir -ErrorAction SilentlyContinue
  Write-Ok "Source build complete ✓"
  $script:ResolvedVersion = $ver
}

# Version symlink (junction on Windows)       
function New-VersionJunction([string]$ver) {
  if ($NoVersionedDir -or $NoSymlink -or [string]::IsNullOrEmpty($ver)) { return }

  $junctionPath = Join-Path $Prefix "current"
  $targetPath = Join-Path $Prefix $ver

  Write-Step "Creating version junction"
  if ($DryRun) { Write-Dry "Junction: $junctionPath -> $targetPath"; return }

  if (Test-Path $junctionPath) {
    Remove-Item $junctionPath -Recurse -Force -ErrorAction SilentlyContinue
  }
  try {
    # On Windows 10 1903+ symlinks work without admin; junctions always work
    New-Item -ItemType Junction -Path $junctionPath -Value $targetPath | Out-Null
    Write-Ok "Junction: $junctionPath → $ver"
  } catch {
    Write-Warn "Could not create junction: $_"
  }
}

# CMAKE_PREFIX_PATH registration         
function Register-CmakePrefixPath([string]$path) {
  if (-not $AddToPath) { return }
  if ($DryRun) { Write-Dry "Set CMAKE_PREFIX_PATH user env += $path"; return }

  $existing = [Environment]::GetEnvironmentVariable("CMAKE_PREFIX_PATH", "User")
  if ($existing -and $existing.Contains($path)) {
    Write-Ok "CMAKE_PREFIX_PATH already contains $path"
    return
  }
  $new = if ($existing) { "$path;$existing" } else { $path }
  [Environment]::SetEnvironmentVariable("CMAKE_PREFIX_PATH", $new, "User")
  Write-Ok "CMAKE_PREFIX_PATH updated (user scope) — restart your shell to apply"
}

# Post-install hints           
function Show-Hints([string]$ver) {
  $dest = Get-EffectivePrefix $ver
  Write-Host ""
  Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Green
  Write-Host "  INSTALL COMPLETE" -ForegroundColor Green
  Write-Host ""
  Write-Host "  Installed to:"
  Write-Host "  $dest" -ForegroundColor Cyan
  Write-Host ""
  Write-Host "  Files:"
  Write-Host "  $dest\include\unified_ml"
  Write-Host "  $dest\lib\unified_ml.dll (shared)"
  Write-Host "  $dest\lib\unified_ml_static.lib (static)"
  Write-Host "  $dest\lib\cmake\unified_ml\"
  Write-Host ""
  Write-Host "  CMake usage:" -ForegroundColor Cyan
  Write-Host "  cmake .. -DCMAKE_PREFIX_PATH=`"$dest`""
  Write-Host "  # CMakeLists.txt:"
  Write-Host "  find_package(unified_ml REQUIRED)"
  Write-Host "  target_link_libraries(my_app PRIVATE unified_ml::unified_ml)"
  Write-Host ""
  Write-Host "  Set permanently:" -ForegroundColor Cyan
  Write-Host "  [Environment]::SetEnvironmentVariable('CMAKE_PREFIX_PATH', '$dest', 'User')"
  Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Green
}

# Main            
$script:ResolvedVersion = $Version

switch ($Mode) {
  "download" {
    if (-not (Install-Prebuilt)) { Write-Fail "Download failed. Use -Mode source." }
  }
  "source" {
    $v = if ($Version -eq "latest") { "1.0.0" } else { $Version }
    Build-FromSource $v
  }
  "auto" {
    if (-not (Install-Prebuilt)) {
    Write-Warn "Prebuilt unavailable — falling back to source build"
    $v = if ($Version -eq "latest") { "1.0.0" } else { $Version }
    Build-FromSource $v
    }
  }
}

$ver = $script:ResolvedVersion
New-VersionJunction $ver
Register-CmakePrefixPath (Get-EffectivePrefix $ver)
Show-Hints $ver
