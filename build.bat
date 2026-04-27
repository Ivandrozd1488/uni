@echo off
REM ═══════════════════════════════════════════════════════════════════
REM  unified_ml — Windows Build Script
REM  Requires: CMake 3.17+, C++ compiler (MSVC, MinGW, or Clang)
REM ═══════════════════════════════════════════════════════════════════

echo.
echo  unified_ml — Building...
echo.

REM Check cmake
where cmake >nul 2>nul
if %ERRORLEVEL% neq 0 (
  echo [ERROR] CMake not found. Install from https://cmake.org/download/
  echo   or: winget install Kitware.CMake
  exit /b 1
)

REM Create build directory
if not exist build mkdir build
cd build

REM Configure
echo [1/3] Configuring...
cmake .. -DCMAKE_BUILD_TYPE=Release
if %ERRORLEVEL% neq 0 (
  echo [ERROR] CMake configuration failed.
  echo.
  echo  If using MSVC, run this from "Developer Command Prompt for VS"
  echo  or "x64 Native Tools Command Prompt"
  echo.
  echo  If using MinGW:
  echo  cmake .. -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release
  echo.
  cd ..
  exit /b 1
)

REM Build
echo [2/3] Compiling...
cmake --build . --config Release --parallel
if %ERRORLEVEL% neq 0 (
  echo [ERROR] Build failed.
  cd ..
  exit /b 1
)

REM Run tests
echo [3/3] Running tests...
echo.

if exist Release\test_autograd.exe (
  Release\test_autograd.exe
) else if exist test_autograd.exe (
  test_autograd.exe
) else (
  echo [WARN] test_autograd not found, skipping tests
)

echo.
echo ═══════════════════════════════════════════════════════════════════
echo  BUILD COMPLETE
echo.
echo  Executables in build/ (or build/Release/ for MSVC):
echo  test_autograd — Unit tests
echo  benchmark   — Performance benchmark
echo  benchmark_hpc — Detailed HPC benchmark
echo.
echo  Run benchmark:
echo  cd build
echo  benchmark.exe
echo  benchmark_hpc.exe
echo ═══════════════════════════════════════════════════════════════════

cd ..
