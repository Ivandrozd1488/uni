#!/usr/bin/env bash
# coverage.sh — build with coverage instrumentation, run all tests, generate report.
#
# Usage:
#   ./coverage.sh              # build + test + report in build-coverage/
#   ./coverage.sh --open       # same, then open HTML report in browser
#   ./coverage.sh --clean      # wipe build-coverage/ first
#
# Requirements:
#   - GCC + gcovr >= 5.0   (pip install gcovr)
#   - cmake >= 3.21, ninja (optional but faster)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build-coverage"
REPORT_DIR="${BUILD_DIR}/coverage"

BOLD='\033[1m'; GREEN='\033[32m'; CYAN='\033[36m'; YELLOW='\033[33m'
RED='\033[31m'; RESET='\033[0m'; DIM='\033[2m'

step() { printf "\n${BOLD}${CYAN}❯ %s${RESET}\n" "$*"; }
ok()   { printf "${GREEN}✔ %s${RESET}\n" "$*"; }
warn() { printf "${YELLOW}⚠ %s${RESET}\n" "$*"; }
die()  { printf "${RED}✘ %s${RESET}\n" "$*"; exit 1; }

# ── Parse args ────────────────────────────────────────────────────────────────
OPEN_REPORT=0
CLEAN=0
for arg in "$@"; do
  case "$arg" in
    --open)  OPEN_REPORT=1 ;;
    --clean) CLEAN=1 ;;
    *)       die "Unknown argument: $arg" ;;
  esac
done

# ── Sanity checks ─────────────────────────────────────────────────────────────
command -v cmake  &>/dev/null || die "cmake not found"
command -v gcovr  &>/dev/null || die "gcovr not found — install with: pip install gcovr"
command -v gcov   &>/dev/null || die "gcov not found (GCC required)"

# Prefer ninja if available
CMAKE_GENERATOR="-G Ninja"
command -v ninja &>/dev/null || CMAKE_GENERATOR=""

# ── Clean ─────────────────────────────────────────────────────────────────────
if [[ "$CLEAN" == "1" ]]; then
  step "Cleaning ${BUILD_DIR}"
  rm -rf "${BUILD_DIR}"
fi

# ── Configure ─────────────────────────────────────────────────────────────────
step "Configuring with coverage flags"
cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" \
  ${CMAKE_GENERATOR} \
  -DCMAKE_BUILD_TYPE=Debug \
  -DUNIFIED_ML_ENABLE_COVERAGE=ON \
  -DUNIFIED_ML_BUILD_TESTS=ON \
  -DUNIFIED_ML_BUILD_BENCHMARKS=OFF \
  -DUNIFIED_ML_ENABLE_AVX=OFF \
  -DUNIFIED_ML_BUILD_SHARED=OFF \
  -DUNIFIED_ML_INSTALL=OFF \
  2>&1 | tail -10
ok "Configured"

# ── Build ─────────────────────────────────────────────────────────────────────
step "Building"
cmake --build "${BUILD_DIR}" --parallel "$(nproc 2>/dev/null || echo 4)"
ok "Build complete"

# ── Reset coverage counters ───────────────────────────────────────────────────
step "Resetting .gcda counters"
find "${BUILD_DIR}" -name "*.gcda" -delete 2>/dev/null || true
ok "Counters reset"

# ── Run tests ─────────────────────────────────────────────────────────────────
step "Running CTest"
cd "${BUILD_DIR}"
ctest --output-on-failure --parallel "$(nproc 2>/dev/null || echo 4)" 2>&1
TEST_EXIT=$?
cd "${SCRIPT_DIR}"
if [[ "$TEST_EXIT" != "0" ]]; then
  warn "Some tests failed (exit $TEST_EXIT) — coverage report will still be generated"
fi
ok "Tests done"

# ── Generate report ───────────────────────────────────────────────────────────
step "Generating coverage report → ${REPORT_DIR}/index.html"
mkdir -p "${REPORT_DIR}"

gcovr \
  --root "${SCRIPT_DIR}" \
  --object-directory "${BUILD_DIR}" \
  --exclude "${SCRIPT_DIR}/tests/.*" \
  --exclude "${SCRIPT_DIR}/examples/.*" \
  --exclude "${SCRIPT_DIR}/benchmark.*\.cpp" \
  --exclude ".*abi\.cpp" \
  --exclude ".*types\.cpp" \
  --exclude ".*tree_node\.cpp" \
  --exclude-unreachable-branches \
  --print-summary \
  --html-details "${REPORT_DIR}/index.html" \
  --txt "${REPORT_DIR}/summary.txt"

ok "Report written to ${REPORT_DIR}/index.html"

# ── Optionally open ───────────────────────────────────────────────────────────
if [[ "$OPEN_REPORT" == "1" ]]; then
  if command -v xdg-open &>/dev/null; then
    xdg-open "${REPORT_DIR}/index.html"
  elif command -v open &>/dev/null; then
    open "${REPORT_DIR}/index.html"
  else
    warn "Cannot open browser — view manually: ${REPORT_DIR}/index.html"
  fi
fi

printf "\n${BOLD}${GREEN}Done.${RESET}  Coverage report: ${DIM}${REPORT_DIR}/index.html${RESET}\n\n"
