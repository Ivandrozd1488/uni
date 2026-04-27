#!/usr/bin/env bash
# build.sh — unified_ml build script

set -uo pipefail # NO -e: we handle errors explicitly

# ANSI palette
if [[ -t 1 ]]; then
  BOLD='\033[1m'  DIM='\033[2m'  RESET='\033[0m'
  GREEN='\033[32m'  CYAN='\033[36m'  YELLOW='\033[33m'
  RED='\033[31m'  GRAY='\033[90m'  WHITE='\033[97m'
  MAGENTA='\033[35m'
else
  BOLD='' DIM='' RESET='' GREEN='' CYAN='' YELLOW='' RED='' GRAY='' WHITE='' MAGENTA=''
fi

step()    { printf "\n${BOLD}${CYAN}  ❯${RESET}${BOLD}  %s${RESET}\n" "$*"; }
ok()      { printf "${GREEN}  ✔${RESET}  %s\n" "$*"; }
info()    { printf "${DIM}   %s${RESET}\n" "$*"; }
warn()    { printf "${YELLOW}  ⚠${RESET}  %s\n" "$*"; }
err()     { printf "${RED}  ✘${RESET}  %s\n" "$*"; }
die()     { err "$*"; exit 1; }
sep()     { printf "${GRAY}        ${RESET}\n"; }
maxinfo() { printf "${MAGENTA}  ★${RESET}  %s\n" "$*"; }

draw_bar() {
  local pct=$1 width=38
  local filled=$(( pct * width / 100 ))
  local empty=$(( width - filled ))
  printf "\r  ${CYAN}[${GREEN}"
  printf "%${filled}s" '' | tr ' ' '▪'
  printf "${GRAY}%${empty}s${CYAN}]${RESET} ${BOLD}%3d%%${RESET}" '' "$pct"
}

# Helpers
cpu_has() { grep -qw "$1" /proc/cpuinfo 2>/dev/null; }

is_ci() {
  [[ -n "${CI:-}" || -n "${GITHUB_ACTIONS:-}" || -n "${GITLAB_CI:-}" || \
   -n "${JENKINS_URL:-}" || -n "${CIRCLECI:-}" || -n "${TRAVIS:-}" || \
   -n "${BUILDKITE:-}" || -n "${TF_BUILD:-}" ]]
}

compiler_supports_flag() {
  echo "int main(){}" | ${CXX:-g++} "$1" -x c++ - -o /dev/null 2>/dev/null
}

detect_simd() {
  if cpu_has avx512f && cpu_has avx512bw && cpu_has avx512dq; then echo avx512
  elif cpu_has avx2;  then echo avx2
  elif cpu_has avx;   then echo avx
  else                     echo none
  fi
}

detect_cuda() {
  command -v nvcc &>/dev/null && nvcc --version &>/dev/null
}

detect_lto() {
  compiler_supports_flag "-flto"
}

# cmake --build с прогресс-баром.
run_build() {
  local jobs=$1 warn_log=$2
  local build_log; build_log=$(mktemp)
  local last_pct=-1

  draw_bar 0

  cmake --build . --config Release -j "$jobs" 2>&1 | tee "$build_log" | \
  while IFS= read -r line; do
    if [[ "$line" =~ ^\[\ *([0-9]+)% ]]; then
      local pct="${BASH_REMATCH[1]}"
      if (( pct != last_pct )); then draw_bar "$pct"; last_pct=$pct; fi
    fi
    [[ "$line" == *"warning:"* ]] && echo "$line" >> "$warn_log"
  done

  local cmake_exit=${PIPESTATUS[0]}

  draw_bar 100
  printf "\n"

  if (( cmake_exit != 0 )); then
    printf "\n"
    err "Compilation failed  (cmake exit $cmake_exit)"
    printf "\n"
    grep -E "error:|undefined reference|ld returned" "$build_log" \
    | head -20 \
    | while IFS= read -r l; do
        printf "  ${DIM}%s${RESET}\n" "$l"
      done
    rm -f "$build_log"
    return 1
  fi

  rm -f "$build_log"
  return 0
}

# ── Parse CLI flags ──────────────────────────────────────────────────────────
EXTRA_CMAKE_ARGS=()
OPT_AVX512=""   # auto | on | off
OPT_NATIVE=""   # auto | on | off
OPT_MAX=0       # --max master switch
OPT_LTO=""      # on | off | ""
OPT_CUDA=""     # on | auto | off | ""
OPT_OPENMP=""   # on | off | ""
JOBS=""

for arg in "$@"; do
  case "$arg" in
    --max)        OPT_MAX=1 ;;
    --avx512)     OPT_AVX512=on  ;;
    --no-avx512)  OPT_AVX512=off ;;
    --native)     OPT_NATIVE=on  ;;
    --no-native)  OPT_NATIVE=off ;;
    --lto)        OPT_LTO=on     ;;
    --no-lto)     OPT_LTO=off    ;;
    --cuda)       OPT_CUDA=on    ;;
    --no-cuda)    OPT_CUDA=off   ;;
    --no-openmp)  OPT_OPENMP=off ;;
    --debug)      EXTRA_CMAKE_ARGS+=("-DCMAKE_BUILD_TYPE=Debug") ;;
    -j*)          JOBS="${arg#-j}" ;;
    --help|-h)
      printf "\nUsage: %s [OPTIONS]\n\n" "$0"
      printf "  ${BOLD}--max${RESET}         Максимальная производительность: всё включается автоматически\n"
      printf "  --avx512      Принудительно включить AVX-512 SIMD\n"
      printf "  --no-avx512   Отключить AVX-512\n"
      printf "  --native      Принудительно включить -march=native\n"
      printf "  --no-native   Отключить -march=native\n"
      printf "  --lto         Включить Link-Time Optimisation\n"
      printf "  --no-lto      Отключить LTO\n"
      printf "  --cuda        Включить CUDA backend (нужен nvcc)\n"
      printf "  --no-cuda     Отключить CUDA\n"
      printf "  --no-openmp   Отключить OpenMP\n"
      printf "  --debug       Debug сборка\n"
      printf "  -jN           Использовать N потоков компиляции\n\n"
      exit 0
      ;;
  esac
done

# ── --max: выставить всё в максимум ─────────────────────────────────────────
if (( OPT_MAX )); then
  [[ -z "$OPT_AVX512" ]] && OPT_AVX512=on
  [[ -z "$OPT_NATIVE" ]] && OPT_NATIVE=on
  [[ -z "$OPT_LTO"    ]] && OPT_LTO=on
  [[ -z "$OPT_CUDA"   ]] && OPT_CUDA=auto   # включить если nvcc доступен
  [[ -z "$OPT_OPENMP" ]] && OPT_OPENMP=on
fi

# ── Banner ───────────────────────────────────────────────────────────────────
printf "\n"
printf "  ${BOLD}${WHITE}unified_ml${RESET}  ${DIM}C++ Machine Learning Library${RESET}\n"
if (( OPT_MAX )); then
  printf "  ${BOLD}${MAGENTA}★ MAX PERFORMANCE BUILD${RESET}\n"
fi
sep

# ── Dependency checks ────────────────────────────────────────────────────────
step "Checking dependencies"

command -v cmake &>/dev/null  || die "CMake not found — sudo apt install cmake"
(command -v g++ &>/dev/null || command -v c++ &>/dev/null) \
  || die "No C++ compiler — sudo apt install g++"

CMAKE_VER=$(cmake --version | awk 'NR==1{print $3}')
CXX_BIN=${CXX:-g++}
CXX_VER=$($CXX_BIN --version 2>/dev/null | awk 'NR==1{print $NF}')
NPROC=${JOBS:-$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 2)}

ok "cmake $CMAKE_VER"
ok "$CXX_BIN $CXX_VER"
ok "$NPROC hardware threads"

# ── SIMD detection ───────────────────────────────────────────────────────────
step "Detecting CPU capabilities"

SIMD_TIER=$(detect_simd)
USED_NATIVE=0

# AVX-512
case "${OPT_AVX512:-auto}" in
  on)
    EXTRA_CMAKE_ARGS+=("-DUNIFIED_ML_ENABLE_AVX512=ON")
    if (( OPT_MAX )); then maxinfo "AVX-512  ENABLED  (--max)"
    else ok "AVX-512  (forced via --avx512)"; fi
    ;;
  off)
    EXTRA_CMAKE_ARGS+=("-DUNIFIED_ML_ENABLE_AVX512=OFF")
    warn "AVX-512  disabled via --no-avx512"
    ;;
  auto)
    if [[ "$SIMD_TIER" == avx512 ]]; then
      EXTRA_CMAKE_ARGS+=("-DUNIFIED_ML_ENABLE_AVX512=ON")
      ok "AVX-512  (avx512f · avx512bw · avx512dq)"
    elif [[ "$SIMD_TIER" == avx2 ]]; then
      ok "AVX2 + FMA  (no AVX-512 on this CPU)"
    else
      ok "SIMD tier: $SIMD_TIER"
    fi
    ;;
esac

# -march=native
case "${OPT_NATIVE:-auto}" in
  on)
    if compiler_supports_flag "-march=native"; then
      EXTRA_CMAKE_ARGS+=("-DUNIFIED_ML_NATIVE_MARCH=ON")
      if (( OPT_MAX )); then maxinfo "-march=native  ENABLED  (--max)"
      else ok "-march=native  (forced)"; fi
      USED_NATIVE=1
    else
      warn "-march=native  не поддерживается компилятором — пропущено"
    fi
    ;;
  off)
    warn "-march=native  disabled via --no-native"
    ;;
  auto)
    if is_ci; then
      info "-march=native  skipped  (CI environment)"
    elif compiler_supports_flag "-march=native"; then
      EXTRA_CMAKE_ARGS+=("-DUNIFIED_ML_NATIVE_MARCH=ON")
      ok "-march=native  (local build — use --no-native to disable)"
      USED_NATIVE=1
    else
      warn "-march=native  not supported by this compiler"
    fi
    ;;
esac

# ── OpenMP ───────────────────────────────────────────────────────────────────
case "${OPT_OPENMP:-}" in
  on)
    EXTRA_CMAKE_ARGS+=("-DUNIFIED_ML_ENABLE_OPENMP=ON")
    if (( OPT_MAX )); then maxinfo "OpenMP  ENABLED  (--max)"
    else ok "OpenMP  enabled"; fi
    ;;
  off)
    EXTRA_CMAKE_ARGS+=("-DUNIFIED_ML_ENABLE_OPENMP=OFF")
    warn "OpenMP  disabled via --no-openmp"
    ;;
  # пусто = CMake default (ON), не трогаем
esac

# ── LTO ──────────────────────────────────────────────────────────────────────
case "${OPT_LTO:-}" in
  on)
    if detect_lto; then
      EXTRA_CMAKE_ARGS+=("-DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON")
      if (( OPT_MAX )); then maxinfo "LTO  ENABLED  (interprocedural optimisation)"
      else ok "LTO  enabled (--lto)"; fi
    else
      warn "LTO  не поддерживается компилятором — пропущено"
    fi
    ;;
  off)
    warn "LTO  disabled via --no-lto"
    ;;
esac

# ── CUDA ─────────────────────────────────────────────────────────────────────
case "${OPT_CUDA:-}" in
  on)
    if detect_cuda; then
      EXTRA_CMAKE_ARGS+=("-DUNIFIED_ML_ENABLE_CUDA=ON")
      CUDA_VER=$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9.]+' || echo "?")
      if (( OPT_MAX )); then maxinfo "CUDA  ENABLED  (nvcc $CUDA_VER)"
      else ok "CUDA  enabled (nvcc $CUDA_VER)"; fi
    else
      warn "CUDA  запрошена, но nvcc не найден — пропущено"
    fi
    ;;
  auto)
    if detect_cuda; then
      EXTRA_CMAKE_ARGS+=("-DUNIFIED_ML_ENABLE_CUDA=ON")
      CUDA_VER=$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9.]+' || echo "?")
      maxinfo "CUDA  ENABLED  (nvcc $CUDA_VER, auto-detected)"
    else
      info "CUDA  не обнаружена (nvcc не найден)"
    fi
    ;;
  off)
    warn "CUDA  disabled via --no-cuda"
    ;;
esac

# ── Configure ────────────────────────────────────────────────────────────────
step "Configuring"

mkdir -p build && cd build

cmake .. -DCMAKE_BUILD_TYPE=Release -Wno-dev "${EXTRA_CMAKE_ARGS[@]}" 2>&1 | \
grep -E "^--|error:" | \
while IFS= read -r line; do
  if [[ "$line" =~ error: ]]; then err "${line#-- }"
  else info "${line#-- }"; fi
done

CMAKE_CFG_EXIT=${PIPESTATUS[0]}
if (( CMAKE_CFG_EXIT != 0 )); then
  die "Configure failed — check errors above"
fi

ok "Ready  (Release, build/)"

# ── Compile ──────────────────────────────────────────────────────────────────
step "Compiling  (${NPROC} threads)"

WARN_LOG=$(mktemp)
if run_build "$NPROC" "$WARN_LOG"; then
  N_WARN=$(sort -u "$WARN_LOG" 2>/dev/null | wc -l)
  if (( N_WARN > 0 )); then
    warn "$N_WARN unique warning(s) — rebuild with VERBOSE=1 for details"
    sort -u "$WARN_LOG" | grep -oE 'warning: [^[]+' | sort -u | head -5 | \
    while IFS= read -r w; do info "$w"; done
  fi
  ok "Compiled  (0 errors)"
else
  rm -f "$WARN_LOG"
  die "Compilation failed — see errors above"
fi
rm -f "$WARN_LOG"

# ── Tests ────────────────────────────────────────────────────────────────────
step "Running tests"

TEST_OUT=$(./test_autograd 2>&1) && TEST_EXIT=0 || TEST_EXIT=$?
PASS=$(printf '%s\n' "$TEST_OUT" | grep -c '\[ *OK *\]'  || true)
FAIL=$(printf '%s\n' "$TEST_OUT" | grep -c '\[ *FAILED *\]' || true)

if (( TEST_EXIT == 0 )); then
  ok "$PASS / $((PASS + FAIL)) tests passed"
else
  warn "$FAIL test(s) failed"
  printf '%s\n' "$TEST_OUT" | grep '\[ *FAILED *\]' | \
    while IFS= read -r l; do printf "  ${RED}  ✘${RESET}  ${DIM}%s${RESET}\n" "$l"; done
fi

# ── Summary ──────────────────────────────────────────────────────────────────
SIMD_LABEL="${SIMD_TIER^^}"
(( USED_NATIVE )) && SIMD_LABEL="$SIMD_LABEL · native"
(( OPT_MAX ))     && SIMD_LABEL="MAX · $SIMD_LABEL"

printf "\n"
sep
printf "\n"
printf "  ${BOLD}${GREEN}Build complete${RESET}  ${DIM}(${SIMD_LABEL})${RESET}\n\n"

col() { printf "  ${DIM}%-20s${RESET}  %s\n" "$1" "$2"; }
col "shared lib"     "build/libunified_ml.so"
col "static lib"     "build/libunified_ml.a"
col "tests"          "build/test_autograd"
col "benchmark"      "build/benchmark"
col "benchmark hpc"  "build/benchmark_hpc"

printf "\n"
if (( OPT_MAX )); then
  maxinfo "OMP_NUM_THREADS=${NPROC} ./benchmark_hpc"
else
  info "cd build && ./benchmark"
  info "OMP_NUM_THREADS=4 ./benchmark_hpc"
fi
printf "\n"
sep
printf "\n"

cd ..