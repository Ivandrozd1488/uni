// benchmark.cpp — Microbenchmarks for unified_ml performance analysis.
//
// Tests: matmul, MLP forward/backward, PINN training step (old vs new engine)
// Build: compiled automatically by CMakeLists.txt as 'benchmark' executable

#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "core/linalg.hpp"
#include "models/pinn/pinn_fast.hpp"

// Timing helper            
using Clock = std::chrono::high_resolution_clock;
using Ms  = std::chrono::duration<double, std::milli>;
using Us  = std::chrono::duration<double, std::micro>;

struct Timer {
  Clock::time_point t0 = Clock::now();
  double elapsed_ms() const { return Ms(Clock::now() - t0).count(); }
  double elapsed_us() const { return Us(Clock::now() - t0).count(); }
  void reset() { t0 = Clock::now(); }
};

// Benchmark runner          
struct BenchResult {
  std::string name;
  double mean_ms;
  double min_ms;
  int  iters;
};

template<typename Fn>
BenchResult bench(const std::string& name, int warmup, int iters, Fn&& fn) {
  // Warmup
  for (int i = 0; i < warmup; ++i) fn();

  Timer t;
  double min_ms = 1e18;
  for (int i = 0; i < iters; ++i) {
    Timer t2;
    fn();
    double ms = t2.elapsed_ms();
    if (ms < min_ms) min_ms = ms;
  }
  double total = t.elapsed_ms();
  return { name, total / iters, min_ms, iters };
}

void print_result(const BenchResult& r) {
  std::cout << std::left << std::setw(50) << r.name
      << " mean=" << std::fixed << std::setprecision(4) << r.mean_ms << "ms"
      << "  min=" << r.min_ms << "ms"
      << "  iters=" << r.iters << "\n";
}

//              
// BENCHMARK 1: Tiled matrix multiplication (linalg::Matrix)
//              
void bench_matmul() {
  std::cout << "\n=== Matrix Multiplication ===\n";

  // 256×256 matmul
  {
    constexpr std::size_t N = 256;
    core::Matrix A(N, N, 1.0 / N), B(N, N, 0.5 / N);
    auto r = bench("matmul 256x256", 3, 20, [&]{ auto C = A * B; (void)C; });
    print_result(r);
    // Theoretical GFLOP: 2 * N³ = 33.6M operations
    double gflops = 2.0 * N * N * N / (r.min_ms * 1e6);
    std::cout << "  => " << std::fixed << std::setprecision(2) << gflops << " GFLOP/s\n";
  }

  // 64×64 matmul (MLP hidden layer typical)
  {
    constexpr std::size_t N = 64;
    core::Matrix A(N, N, 1.0 / N), B(N, N, 0.5 / N);
    auto r = bench("matmul 64x64", 5, 100, [&]{ auto C = A * B; (void)C; });
    print_result(r);
  }

  // Matrix-vector multiply 512×64 (typical MLP layer weight × input batch)
  {
    constexpr std::size_t M = 512, K = 64;
    core::Matrix A(M, K, 0.1);
    core::Vector v(K, 1.0);
    auto r = bench("matvec 512x64", 5, 100, [&]{ auto y = A * v; (void)y; });
    print_result(r);
  }
}

//              
// BENCHMARK 2: PINN Fast Engine — Poisson 1D
//              
void bench_pinn_fast() {
  std::cout << "\n=== PINN Fast Engine (analytical grad, no dynamic graph) ===\n";

  // Architecture: 1→64→32→1
  pinn::PinnFastNet net({1, 64, 32, 1}, 42, 1e-3);

  // 100 collocation points, 2 BC points
  constexpr double PI = 3.14159265358979;
  constexpr int N_COLL = 100;
  constexpr int N_BC = 2;

  std::vector<double> coll_x(N_COLL), f_vals(N_COLL);
  for (int i = 0; i < N_COLL; ++i) {
    coll_x[i] = (i + 1.0) / (N_COLL + 1);
    f_vals[i] = PI * PI * std::sin(PI * coll_x[i]);  // -u'' = f
  }
  std::vector<double> bc_x  = {0.0, 1.0};
  std::vector<double> bc_u  = {0.0, 0.0};

  // Single training step
  {
    auto r = bench("PINN fast train_step (100 coll, 2 BC)", 3, 50, [&]{
    net.train_step_poisson1d(coll_x, f_vals, bc_x, bc_u);
    });
    print_result(r);
  }

  // Inference only (predict 1000 points)
  {
    std::vector<double> test_x(1000);
    for (int i = 0; i < 1000; ++i) test_x[i] = i / 999.0;
    auto r = bench("PINN fast predict (1000 points)", 3, 50, [&]{
    auto preds = net.predict(test_x);
    (void)preds;
    });
    print_result(r);
  }

  // Full training run: 1000 iterations
  {
    pinn::PinnFastNet net2({1, 64, 32, 1}, 42, 1e-3);
    Timer t;
    for (int i = 0; i < 1000; ++i)
    net2.train_step_poisson1d(coll_x, f_vals, bc_x, bc_u);
    double ms = t.elapsed_ms();
    std::cout << std::left << std::setw(50) << "PINN fast 1000 iters (end-to-end)"
      << " total=" << std::fixed << std::setprecision(1) << ms << "ms"
      << "  per_iter=" << ms / 1000 << "ms\n";

    // Check accuracy
    double max_err = 0.0;
    for (int i = 1; i < 20; ++i) {
    double x = i / 20.0;
    double pred = net2.forward_scalar(&x);
    double exact = std::sin(PI * x);
    max_err = std::max(max_err, std::abs(pred - exact));
    }
    std::cout << "  => max L∞ error after 1000 iters: " << max_err << "\n";
  }
}

//              
// BENCHMARK 3: Aligned memory allocation vs std::vector
//              
void bench_memory() {
  std::cout << "\n=== Memory / Alignment Microbenchmarks ===\n";
  constexpr std::size_t N = 1024 * 1024;  // 8 MB of doubles

  // AlignedVec dot product
  {
    core::AlignedVec a(N, 1.0), b(N, 2.0);
    auto r = bench("AlignedVec dot (1M doubles)", 3, 20, [&]{
    double s = 0.0;
#pragma omp simd reduction(+:s)
    for (std::size_t i = 0; i < N; ++i) s += a[i] * b[i];
    volatile double v = s; (void)v;
    });
    print_result(r);
  }

  // std::vector dot product (likely unaligned)
  {
    std::vector<double> a(N, 1.0), b(N, 2.0);
    auto r = bench("std::vector dot (1M doubles)", 3, 20, [&]{
    double s = 0.0;
    for (std::size_t i = 0; i < N; ++i) s += a[i] * b[i];
    volatile double v = s; (void)v;
    });
    print_result(r);
  }
}

//              
// BENCHMARK 4: Cache-blocking effect on transpose
//              
void bench_transpose() {
  std::cout << "\n=== Matrix Transpose (cache-blocking) ===\n";

  constexpr std::size_t N = 1024;
  core::Matrix A(N, N);
  for (std::size_t i = 0; i < N; ++i)
    for (std::size_t j = 0; j < N; ++j)
    A(i, j) = static_cast<double>(i * N + j);

  auto r = bench("tiled transpose 1024x1024", 3, 20, [&]{
    auto B = A.transpose(); (void)B;
  });
  print_result(r);
  double gb = 2.0 * N * N * 8.0 / 1e9; // read + write
  double bw = gb / (r.min_ms / 1000.0);
  std::cout << "  => " << std::fixed << std::setprecision(1) << bw << " GB/s effective bandwidth\n";
}

//              
// BENCHMARK 5: Activation functions (SIMD)
//              
void bench_activations() {
  std::cout << "\n=== Activation Functions (SIMD) ===\n";
  constexpr std::size_t N = 65536;

  std::vector<double> x_data(N), out(N);
  for (std::size_t i = 0; i < N; ++i) x_data[i] = (i / double(N)) * 4.0 - 2.0;

  {
    auto r = bench("tanh SIMD 65536 elements", 5, 100, [&]{
#pragma omp simd
    for (std::size_t i = 0; i < N; ++i) out[i] = std::tanh(x_data[i]);
    volatile double v = out[0]; (void)v;
    });
    print_result(r);
    double gelems = N / (r.min_ms * 1e6);
    std::cout << "  => " << std::fixed << std::setprecision(2) << gelems << " Gelem/s\n";
  }

  {
    auto r = bench("tanh backward (sech²) 65536 elems", 5, 100, [&]{
    const double* xp = x_data.data();
    double* op = out.data();
#pragma omp simd
    for (std::size_t i = 0; i < N; ++i) {
      double t = std::tanh(xp[i]);
      op[i] = 1.0 - t * t;
    }
    volatile double v = out[0]; (void)v;
    });
    print_result(r);
  }
}

//              
// BENCHMARK 6: PINN forward derivatives (analytical vs baseline cost)
//              
void bench_pinn_forward_derivs() {
  std::cout << "\n=== PINN Forward Derivatives (single point) ===\n";

  pinn::PinnFastNet net({1, 64, 32, 1}, 42, 1e-3);

  // Single-point forward_derivs_1d (fills all caches)
  {
    auto r = bench("forward_derivs_1d (1→64→32→1)", 5, 10000, [&]{
    double x = 0.5;
    // forward_scalar uses forward_and_cache
    volatile double u = net.forward_scalar(&x);
    (void)u;
    });
    print_result(r);
    std::cout << "  => " << std::fixed << std::setprecision(3)
      << r.min_ms * 1000.0 << " μs / inference call\n";
  }
}

int main() {
  std::cout << "=======================================================\n";
  std::cout << " unified_ml Performance Benchmark\n";
  std::cout << "=======================================================\n";
  std::cout << "Build: -O3 -march=native -ffast-math -fopenmp-simd\n\n";

  bench_matmul();
  bench_memory();
  bench_transpose();
  bench_activations();
  bench_pinn_forward_derivs();
  bench_pinn_fast();

  std::cout << "\n=== Done ===\n";
  return 0;
}
