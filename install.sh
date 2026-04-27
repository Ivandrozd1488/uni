#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
#  unified_ml — Production-oriented installer  (Linux / macOS)
#  Version: 1.0.0
#
#  Builds a compatible release configuration with optional internal
#  acceleration backends auto-detected and enabled when available.
#
#  Usage:
#    ./install.sh                          # build & install to /usr/local
#    ./install.sh --prefix ~/.local        # no-sudo user install
#    ./install.sh --prefix /opt/unified_ml # custom path
#    ./install.sh --no-native              # portable binary (no local tuning)
#    ./install.sh --cuda                   # force-enable optional CUDA backend
#    ./install.sh --python                 # force-enable bootstrap Python bindings
#    ./install.sh --dry-run                # preview what would happen
#    ./install.sh --help                   # show this message
#
#  Auto-detected (no flags needed):
#    AVX-512       — optional internal acceleration if supported
#    AVX2+FMA      — optional internal acceleration on supported x86_64 CPUs
#    OpenMP        — optional parallel backend if supported by the compiler
#    -march=native — local-build tuning only
#    CUDA          — optional backend if nvcc is found in PATH
#    Python        — bootstrap bindings if pybind11 is discoverable
# ══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

# ── Colour palette ───────────────────────────────────────────────────────────
if [[ -t 1 ]]; then
  BOLD='\033[1m'  DIM='\033[2m'   RESET='\033[0m'
  GREEN='\033[32m' CYAN='\033[36m' YELLOW='\033[33m'
  RED='\033[31m'  GRAY='\033[90m'  WHITE='\033[97m'
else
  BOLD='' DIM='' RESET='' GREEN='' CYAN='' YELLOW=''
  RED='' GRAY='' WHITE=''
fi

# ── Logging ──────────────────────────────────────────────────────────────────
nl()   { printf '\n'; }
log()  { printf "${GREEN}  ✔${RESET}  %s\n" "$*"; }
info() { printf "${DIM}     %s${RESET}\n"   "$*"; }
warn() { printf "${YELLOW}  ⚠${RESET}  %s\n" "$*" >&2; }
err()  { printf "${RED}  ✘${RESET}  %s\n"   "$*" >&2; exit 1; }
step() { nl; printf "${BOLD}${CYAN}  ❯${RESET}${BOLD}  %s${RESET}\n" "$*"; }
sep()  { printf "${GRAY}  ────────────────────────────────────────${RESET}\n"; }
dry()  { printf "${DIM}  [dry-run] %s${RESET}\n" "$*"; }
feat() { printf "  ${GREEN}✔${RESET}  %-22s ${DIM}%s${RESET}\n" "$1" "$2"; }
miss() { printf "  ${GRAY}–${RESET}  %-22s ${DIM}%s${RESET}\n" "$1" "$2"; }

# ── Progress bar ─────────────────────────────────────────────────────────────
draw_bar() {
  local pct=$1 width=40
  local filled=$(( pct * width / 100 ))
  local empty=$(( width - filled ))
  printf "\r  ${CYAN}[${GREEN}"
  printf "%${filled}s" '' | tr ' ' '▪'
  printf "${GRAY}%${empty}s${CYAN}]${RESET} ${BOLD}%3d%%${RESET}" '' "$pct"
}

# cmake build with live progress bar + error extraction
run_build() {
  local jobs=$1
  local build_log; build_log=$(mktemp)
  local last_pct=-1
  draw_bar 0
  cmake --build . --config Release -j "$jobs" 2>&1 | tee "$build_log" | \
  while IFS= read -r line; do
    if [[ "$line" =~ ^\[\ *([0-9]+)% ]]; then
      local pct="${BASH_REMATCH[1]}"
      if (( pct != last_pct )); then draw_bar "$pct"; last_pct=$pct; fi
    fi
  done
  local cmake_exit=${PIPESTATUS[0]}
  draw_bar 100; nl; nl
  if (( cmake_exit != 0 )); then
    err "Compilation failed (cmake exit $cmake_exit)"
    grep -E "error:|undefined reference|ld returned" "$build_log" | head -20 | \
      while IFS= read -r l; do printf "  ${DIM}%s${RESET}\n" "$l"; done
    rm -f "$build_log"; return 1
  fi
  rm -f "$build_log"; return 0
}

# ── Helpers ──────────────────────────────────────────────────────────────────
cpu_has()                { grep -qw "$1" /proc/cpuinfo 2>/dev/null; }
compiler_supports_flag() { echo "int main(){}" | ${CXX:-g++} "$1" -x c++ - -o /dev/null 2>/dev/null; }
detect_simd() {
  if cpu_has avx512f && cpu_has avx512bw && cpu_has avx512dq; then echo avx512
  elif cpu_has avx2; then echo avx2
  elif cpu_has avx;  then echo avx
  else                    echo none
  fi
}

# ── Defaults ─────────────────────────────────────────────────────────────────
PREFIX="/usr/local"
OPT_NATIVE="auto"   # auto | on | off
OPT_CUDA="auto"     # auto | on | off
OPT_PYTHON="auto"   # auto | on | off
OPT_TESTS="on"
KEEP_ARTIFACTS=0
DRY_RUN=0

# ── Argument parsing ─────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --prefix)     PREFIX="$2";       shift 2 ;;
    --prefix=*)   PREFIX="${1#*=}";  shift   ;;
    --native)     OPT_NATIVE="on";   shift   ;;
    --no-native)  OPT_NATIVE="off";  shift   ;;
    --cuda)       OPT_CUDA="on";     shift   ;;
    --no-cuda)    OPT_CUDA="off";    shift   ;;
    --python)     OPT_PYTHON="on";   shift   ;;
    --no-python)  OPT_PYTHON="off";  shift   ;;
    --no-tests)   OPT_TESTS="off";   shift   ;;
    --keep)       KEEP_ARTIFACTS=1;  shift   ;;
    --dry-run)    DRY_RUN=1;         shift   ;;
    --help|-h)    sed -n '4,19p' "$0"; exit 0 ;;
    *) warn "Unknown argument: $1"; shift ;;
  esac
done

# ── Banner ────────────────────────────────────────────────────────────────────
nl
printf "  ${BOLD}${WHITE}unified_ml${RESET}  ${DIM}HPC Machine Learning Engine — Max-Performance Build${RESET}\n"
sep
printf "  ${DIM}Install prefix : ${CYAN}%s${RESET}\n" "$PREFIX"
[[ $DRY_RUN -eq 1 ]] && printf "  ${YELLOW}  DRY RUN — no changes will be made${RESET}\n"
nl

# ── Platform ──────────────────────────────────────────────────────────────────
step "Detecting platform"
OS="$(uname -s)"; ARCH="$(uname -m)"
case "$OS" in
  Linux)  PLATFORM="linux"  ;;
  Darwin) PLATFORM="macos"  ;;
  *) err "Unsupported OS: $OS  (use install.bat / install.ps1 on Windows)" ;;
esac
case "$ARCH" in
  x86_64)        ARCH_TAG="x86_64" ;;
  arm64|aarch64) ARCH_TAG="arm64"  ;;
  *)             ARCH_TAG="$ARCH"  ;;
esac
log "Platform: ${PLATFORM}-${ARCH_TAG}"

# ── Dependency check ──────────────────────────────────────────────────────────
step "Checking dependencies"
command -v cmake &>/dev/null || err "CMake not found.
  Ubuntu/Debian : sudo apt install cmake build-essential
  macOS         : brew install cmake"

(command -v g++ &>/dev/null || command -v c++ &>/dev/null || \
 command -v clang++ &>/dev/null) || \
  err "No C++ compiler found.
  Ubuntu/Debian : sudo apt install build-essential
  macOS         : xcode-select --install"

CMAKE_VER=$(cmake --version | awk 'NR==1{print $3}')
CXX_BIN=${CXX:-g++}; command -v "$CXX_BIN" &>/dev/null || CXX_BIN=c++
CXX_VER=$($CXX_BIN --version 2>/dev/null | awk 'NR==1{print $NF}')
NPROC=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

# Check CMake >= 3.17
cmake_maj="${CMAKE_VER%%.*}"; cmake_min="${CMAKE_VER#*.}"; cmake_min="${cmake_min%%.*}"
{ (( cmake_maj > 3 )) || (( cmake_maj == 3 && cmake_min >= 17 )); } || \
  err "CMake >= 3.17 required (found ${CMAKE_VER})"

log "cmake ${CMAKE_VER}"
log "${CXX_BIN} ${CXX_VER}"
log "${NPROC} hardware threads"

# ── Hardware capability detection ─────────────────────────────────────────────
step "Detecting hardware capabilities"
CMAKE_ARGS=()
AVX512_ENABLED=0; NATIVE_ENABLED=0; CUDA_ENABLED=0; PYTHON_ENABLED=0

# SIMD — x86_64
if [[ "$ARCH_TAG" == "x86_64" ]]; then
  SIMD_TIER=$(detect_simd)
  CMAKE_ARGS+=("-DUNIFIED_ML_ENABLE_AVX=ON")
  feat "AVX2 + FMA" "baseline x86_64 SIMD (Haswell 2013+)"

  if [[ "$SIMD_TIER" == "avx512" ]]; then
    CMAKE_ARGS+=("-DUNIFIED_ML_ENABLE_AVX512=ON")
    feat "AVX-512" "avx512f · avx512bw · avx512dq detected"
    AVX512_ENABLED=1
  else
    CMAKE_ARGS+=("-DUNIFIED_ML_ENABLE_AVX512=OFF")
    miss "AVX-512" "not available on this CPU (tier: $SIMD_TIER)"
  fi
else
  CMAKE_ARGS+=("-DUNIFIED_ML_ENABLE_AVX=OFF" "-DUNIFIED_ML_ENABLE_AVX512=OFF")
  [[ "$ARCH_TAG" == "arm64" ]] && feat "ARM NEON" "native on Apple Silicon / ARM64"
fi

# -march=native
case "$OPT_NATIVE" in
  on)
    CMAKE_ARGS+=("-DUNIFIED_ML_NATIVE_MARCH=ON")
    feat "-march=native" "forced on"
    NATIVE_ENABLED=1
    ;;
  off)
    CMAKE_ARGS+=("-DUNIFIED_ML_NATIVE_MARCH=OFF")
    miss "-march=native" "disabled via --no-native"
    ;;
  auto)
    if compiler_supports_flag "-march=native"; then
      CMAKE_ARGS+=("-DUNIFIED_ML_NATIVE_MARCH=ON")
      feat "-march=native" "enabled (use --no-native for portable binary)"
      NATIVE_ENABLED=1
    else
      CMAKE_ARGS+=("-DUNIFIED_ML_NATIVE_MARCH=OFF")
      miss "-march=native" "compiler does not support it"
    fi
    ;;
esac

# OpenMP
if compiler_supports_flag "-fopenmp"; then
  CMAKE_ARGS+=("-DUNIFIED_ML_ENABLE_OPENMP=ON")
  feat "OpenMP" "multi-threading enabled"
else
  CMAKE_ARGS+=("-DUNIFIED_ML_ENABLE_OPENMP=OFF")
  miss "OpenMP" "not supported by this compiler"
fi

# CUDA
case "$OPT_CUDA" in
  on)
    CMAKE_ARGS+=("-DUNIFIED_ML_ENABLE_CUDA=ON")
    feat "CUDA" "forced via --cuda"
    CUDA_ENABLED=1
    ;;
  off)
    CMAKE_ARGS+=("-DUNIFIED_ML_ENABLE_CUDA=OFF")
    miss "CUDA" "disabled via --no-cuda"
    ;;
  auto)
    if command -v nvcc &>/dev/null; then
      NVCC_VER=$(nvcc --version 2>/dev/null | grep "release" | awk '{print $5}' | tr -d ',')
      CMAKE_ARGS+=("-DUNIFIED_ML_ENABLE_CUDA=ON")
      feat "CUDA" "nvcc ${NVCC_VER} found"
      CUDA_ENABLED=1
    else
      CMAKE_ARGS+=("-DUNIFIED_ML_ENABLE_CUDA=OFF")
      miss "CUDA" "nvcc not in PATH (install CUDA toolkit or use --cuda)"
    fi
    ;;
esac

# Python bindings
case "$OPT_PYTHON" in
  on)
    CMAKE_ARGS+=("-DUNIFIED_ML_BUILD_PYTHON=ON")
    feat "Python bindings" "forced via --python"
    PYTHON_ENABLED=1
    ;;
  off)
    CMAKE_ARGS+=("-DUNIFIED_ML_BUILD_PYTHON=OFF")
    miss "Python bindings" "disabled via --no-python"
    ;;
  auto)
    if python3 -c "import pybind11" 2>/dev/null || \
       pkg-config --exists pybind11 2>/dev/null; then
      CMAKE_ARGS+=("-DUNIFIED_ML_BUILD_PYTHON=ON")
      feat "Python bindings" "pybind11 found"
      PYTHON_ENABLED=1
    else
      CMAKE_ARGS+=("-DUNIFIED_ML_BUILD_PYTHON=OFF")
      miss "Python bindings" "pybind11 not found (pip install pybind11 to enable)"
    fi
    ;;
esac

# Both library forms always
CMAKE_ARGS+=("-DUNIFIED_ML_BUILD_SHARED=ON" "-DUNIFIED_ML_BUILD_STATIC=ON")
feat "Shared library" ".so / .dylib"
feat "Static library" ".a"

# Tests + benchmarks
if [[ "$OPT_TESTS" == "on" ]]; then
  CMAKE_ARGS+=("-DUNIFIED_ML_BUILD_TESTS=ON" "-DUNIFIED_ML_BUILD_BENCHMARKS=ON")
  feat "Tests + benchmarks" "built alongside library"
else
  CMAKE_ARGS+=("-DUNIFIED_ML_BUILD_TESTS=OFF" "-DUNIFIED_ML_BUILD_BENCHMARKS=OFF")
fi

# ── Configure ─────────────────────────────────────────────────────────────────
step "Configuring build"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd || echo "$PWD")"
BUILD_DIR="${SCRIPT_DIR}/_build_$$"

if [[ $DRY_RUN -eq 1 ]]; then
  dry "cmake -S $SCRIPT_DIR -B $BUILD_DIR -DCMAKE_BUILD_TYPE=Release \\"
  for arg in "${CMAKE_ARGS[@]}"; do dry "  $arg \\"; done
  dry "  -DCMAKE_INSTALL_PREFIX=$PREFIX"
  dry "cmake --build $BUILD_DIR -j $NPROC"
  dry "cmake --install $BUILD_DIR"
  nl; log "Dry run complete — no changes made"; nl; exit 0
fi

cmake -S "$SCRIPT_DIR" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" \
  -DUNIFIED_ML_INSTALL=ON \
  -DUNIFIED_ML_VERSIONED_INSTALL=OFF \
  -Wno-dev \
  "${CMAKE_ARGS[@]}" \
  2>&1 | grep -E "^--|error:|warning:" | \
  while IFS= read -r line; do
    case "$line" in
      *error:*)   printf "  ${RED}%s${RESET}\n"    "${line#-- }" ;;
      *warning:*) printf "  ${YELLOW}%s${RESET}\n" "${line#-- }" ;;
      *)          info "${line#-- }" ;;
    esac
  done

CMAKE_CFG_EXIT=${PIPESTATUS[0]}
(( CMAKE_CFG_EXIT != 0 )) && err "Configure failed — see errors above"
log "Configuration complete"

# ── Compile ───────────────────────────────────────────────────────────────────
step "Compiling  (${NPROC} threads)"
cd "$BUILD_DIR"
if run_build "$NPROC"; then
  log "Compiled successfully"
else
  [[ $KEEP_ARTIFACTS -eq 0 ]] && rm -rf "$BUILD_DIR"
  err "Compilation failed"
fi

# ── Tests ─────────────────────────────────────────────────────────────────────
if [[ "$OPT_TESTS" == "on" ]] && [[ -f "./test_autograd" ]]; then
  step "Running smoke tests"
  TEST_OUT=$(./test_autograd 2>&1) && TEST_EXIT=0 || TEST_EXIT=$?
  PASS=$(printf '%s\n' "$TEST_OUT" | grep -c '\[ *OK *\]'     || true)
  FAIL=$(printf '%s\n' "$TEST_OUT" | grep -c '\[ *FAILED *\]' || true)
  if (( TEST_EXIT == 0 )); then
    log "$PASS / $((PASS + FAIL)) tests passed"
  else
    warn "$FAIL test(s) failed — library still installed"
    printf '%s\n' "$TEST_OUT" | grep '\[ *FAILED *\]' | \
      while IFS= read -r l; do printf "  ${RED}  ✘${RESET}  ${DIM}%s${RESET}\n" "$l"; done
  fi
fi

# ── Install ───────────────────────────────────────────────────────────────────
step "Installing to ${PREFIX}"
if [[ -w "$(dirname "$PREFIX")" ]] || [[ -w "$PREFIX" ]]; then
  mkdir -p "$PREFIX"
  cmake --install .
else
  warn "No write permission — using sudo"
  sudo mkdir -p "$PREFIX"
  sudo cmake --install .
fi
log "Installed ✓"

cd "$SCRIPT_DIR"
[[ $KEEP_ARTIFACTS -eq 0 ]] && rm -rf "$BUILD_DIR"

# ── Summary ───────────────────────────────────────────────────────────────────
nl; sep; nl
FEAT_LIST=""
(( AVX512_ENABLED )) && FEAT_LIST="AVX-512" || FEAT_LIST="AVX2+FMA"
(( NATIVE_ENABLED )) && FEAT_LIST="${FEAT_LIST} · native"
(( CUDA_ENABLED   )) && FEAT_LIST="${FEAT_LIST} · CUDA"
(( PYTHON_ENABLED )) && FEAT_LIST="${FEAT_LIST} · Python"
printf "  ${BOLD}${GREEN}Install complete${RESET}  ${DIM}(%s)${RESET}\n\n" "$FEAT_LIST"

col() { printf "  ${DIM}%-26s${RESET}  %s\n" "$1" "$2"; }
col "prefix"       "${PREFIX}"
col "headers"      "${PREFIX}/include/unified_ml"
col "shared lib"   "${PREFIX}/lib/libunified_ml.so"
col "static lib"   "${PREFIX}/lib/libunified_ml.a"
col "CMake config" "${PREFIX}/lib/cmake/unified_ml/"
col "pkg-config"   "${PREFIX}/lib/pkgconfig/unified_ml.pc"
nl
printf "  ${BOLD}Quick start — CMake:${RESET}\n"
printf "  ${DIM}find_package(unified_ml REQUIRED)\n"
printf "  target_link_libraries(my_app PRIVATE unified_ml::unified_ml)${RESET}\n"
nl
printf "  ${BOLD}Quick start — pkg-config:${RESET}\n"
printf "  ${DIM}g++ main.cpp \$(pkg-config --cflags --libs unified_ml) -o app${RESET}\n"
nl
if [[ "$PREFIX" != "/usr/local" && "$PREFIX" != "/usr" ]]; then
  printf "  ${YELLOW}Non-standard prefix — add to ~/.bashrc or ~/.zshrc:${RESET}\n"
  printf "  export CMAKE_PREFIX_PATH=\"%s:\$CMAKE_PREFIX_PATH\"\n" "$PREFIX"
  printf "  export PKG_CONFIG_PATH=\"%s/lib/pkgconfig:\$PKG_CONFIG_PATH\"\n" "$PREFIX"
  [[ "$PLATFORM" == "linux" ]] && \
    printf "  export LD_LIBRARY_PATH=\"%s/lib:\$LD_LIBRARY_PATH\"\n" "$PREFIX"
  [[ "$PLATFORM" == "macos" ]] && \
    printf "  export DYLD_LIBRARY_PATH=\"%s/lib:\$DYLD_LIBRARY_PATH\"\n" "$PREFIX"
  nl
fi
sep; nl
