@echo off
:: ══════════════════════════════════════════════════════════════════════════════
::  unified_ml — Windows installer (batch)
::  For a better experience use install.ps1 (supports prebuilt download).
::
::  Usage:
::  install.bat         → install to C:\unified_ml
::  install.bat --prefix D:\libs  → custom prefix
::
::  Requires: cmake in PATH
::    Run from VS Developer Command Prompt for best results
:: ══════════════════════════════════════════════════════════════════════════════
setlocal enabledelayedexpansion

set "PREFIX=C:\unified_ml"

:parse
if "%~1"=="" goto args_done
if /i "%~1"=="--prefix" ( set "PREFIX=%~2" & shift & shift & goto parse )
shift & goto parse
:args_done

echo.
echo unified_ml Installer ^(Windows^)
echo Install prefix: %PREFIX%
echo.

cmake --version >nul 2>&1 || (
  echo [ERROR] cmake not found. Download: https://cmake.org/download/
  exit /b 1
)

set "BUILD_DIR=%~dp0build_install_win"

echo [1/3] Configuring...
cmake -S "%~dp0." -B "%BUILD_DIR%" ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_INSTALL_PREFIX="%PREFIX%" ^
  -DUNIFIED_ML_BUILD_TESTS=OFF ^
  -DUNIFIED_ML_BUILD_BENCHMARKS=OFF ^
  -DUNIFIED_ML_INSTALL=ON ^
  -DUNIFIED_ML_ENABLE_OPENMP=ON ^
  -DUNIFIED_ML_ENABLE_AVX=ON
if errorlevel 1 ( echo [ERROR] Configure failed & exit /b 1 )

echo [2/3] Compiling...
cmake --build "%BUILD_DIR%" --config Release --parallel
if errorlevel 1 ( echo [ERROR] Build failed & exit /b 1 )

echo [3/3] Installing to %PREFIX% ...
cmake --install "%BUILD_DIR%" --config Release
if errorlevel 1 ( echo [ERROR] Install failed & exit /b 1 )

rmdir /s /q "%BUILD_DIR%" 2>nul

echo.
echo ════════════════════════════════════════════════════════════════
echo INSTALL COMPLETE
echo.
echo Header  : %PREFIX%\include\unified_ml
echo Library : %PREFIX%\lib\unified_ml.lib
echo CMake : %PREFIX%\lib\cmake\unified_ml\
echo.
echo Use in CMake:
echo   cmake .. -DCMAKE_PREFIX_PATH="%PREFIX%"
echo   find_package^(unified_ml REQUIRED^)
echo   target_link_libraries^(my_app PRIVATE unified_ml::unified_ml^)
echo ════════════════════════════════════════════════════════════════
