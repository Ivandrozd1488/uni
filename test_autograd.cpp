// test_autograd.cpp — standalone regression tests for the four fixed bugs.
//
// Build (from project root):
// g++ -std=c++17 -O2 -Wall -Iinclude \
//   src/autograd/tensor.cpp src/autograd/ops.cpp src/autograd/functional.cpp \
//   test_autograd.cpp -o test_autograd && ./test_autograd
//
// Each test prints PASS / FAIL.  Exit code 0 iff all pass.

#include "autograd/autograd.h"
#include "core/activations.hpp"
#include "models/mlp/loss.hpp"
// ops free functions (mul, sum, matmul, …) are declared in tensor.h via forward
// declarations or resolved at link time — no separate ops.h header exists.

#include <cmath>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <cstdio>

using namespace autograd;

// helpers            

static int g_pass = 0, g_fail = 0;

static void check(bool ok, const std::string& name)
{
  if (ok) { std::cout << "  PASS  " << name << "\n"; ++g_pass; }
  else  { std::cout << "  FAIL  " << name << "\n"; ++g_fail; }
}

static bool near(double a, double b, double tol = 1e-9) { return std::abs(a - b) <= tol; }

static bool gradcheck(const std::string& name,
                      const std::function<Tensor(const Tensor&)>& f,
                      const Tensor& x,
                      double tol = 1e-5)
{
  Tensor xv(x.data(), x.shape(), true);
  Tensor y = f(xv);
  Tensor ga = functional::grad(y, xv, false, false);
  Tensor gn = functional::numerical_grad(f, xv, 1e-6);
  double max_abs = 0.0;
  for (std::size_t i = 0; i < ga.numel(); ++i)
    max_abs = std::max(max_abs, std::abs(ga.data()[i] - gn.data()[i]));
  check(max_abs <= tol, name + " (max_abs_err=" + std::to_string(max_abs) + ")");
  return max_abs <= tol;
}

// ══════════════════════════════════════════════════════════════════════════════
//  Bug 1 — mul lacked broadcast support
//
//  C = A * B  with A shape [3] and B shape [1].
//  Expected forward:  C[i] = A[i] * B[0]  = {2, 6, 12}
//  Expected backward:
//  dA[i]  = dC[i] * B[0]   = {1, 1, 1} * 2  = {2, 2, 2}
//  dB[0]  = Σ dC[i] * A[i]   = 1*1 + 1*2 + 1*3 = 6
// ══════════════════════════════════════════════════════════════════════════════

static void test_mul_broadcast()
{
  std::cout << "\n[Bug 1] mul broadcast\n";

  Tensor a({1.0, 2.0, 3.0}, {3}, /*req_grad=*/true);
  Tensor b({2.0},    {1}, /*req_grad=*/true);

  // Forward
  bool threw = false;
  Tensor c;
  try { c = mul(a, b); }
  catch (...) { threw = true; }

  check(!threw,    "mul broadcast does not throw");
  check(c.numel() == 3, "output numel == 3");
  if (!threw && c.numel() == 3) {
    check(near(c.data()[0], 2.0), "c[0] == 2");
    check(near(c.data()[1], 4.0), "c[1] == 4");
    check(near(c.data()[2], 6.0), "c[2] == 6");
  }

  // Backward (fast path)
  if (!threw && c.numel() == 3) {
    Tensor s = sum(c);
    s.backward();
    check(near(a.grad()[0], 2.0) && near(a.grad()[1], 2.0) && near(a.grad()[2], 2.0),
      "grad_A == {2,2,2}");
    check(near(b.grad()[0], 6.0), "grad_B == {6}");
  }

  // Backward via VJP (create_graph path)
  {
    Tensor a2({1.0, 2.0, 3.0}, {3}, true);
    Tensor b2({2.0},    {1}, true);
    Tensor c2 = sum(mul(a2, b2));
    Tensor ga = functional::grad(c2, a2, /*retain=*/false, /*create_graph=*/true);
    Tensor gb = functional::grad(c2, b2, false, true);
    check(ga.numel() == 3, "VJP grad_A numel == 3");
    check(near(ga.data()[0], 2.0) && near(ga.data()[1], 2.0) && near(ga.data()[2], 2.0),
      "VJP grad_A == {2,2,2}");
    check(near(gb.data()[0], 6.0), "VJP grad_B == {6}");
  }
}

// ══════════════════════════════════════════════════════════════════════════════
//  Bug 2 — vjp_fn cast UB  (std::any_cast<GradMap&> → UB on type mismatch)
//
//  After the fix, a badly-typed return triggers a clear runtime_error rather
//  than UB.  We verify the happy path (mul VJP works) and that the fix compiles
//  (the pointer-form cast is used).
// ══════════════════════════════════════════════════════════════════════════════

static void test_vjp_cast_safety()
{
  std::cout << "\n[Bug 2] vjp_fn any_cast safety\n";

  Tensor x({1.0, 2.0, 3.0}, {3}, true);
  Tensor y({4.0, 5.0, 6.0}, {3}, true);
  Tensor z = sum(mul(x, y));   // z = 1*4 + 2*5 + 3*6 = 32

  check(near(z.item(), 32.0), "forward value correct");

  Tensor gx = functional::grad(z, x, false, /*create_graph=*/true);
  Tensor gy = functional::grad(z, y, false, true);

  // dz/dx[i] = y[i]
  check(near(gx.data()[0], 4.0) && near(gx.data()[1], 5.0) && near(gx.data()[2], 6.0),
    "grad_x == y via create_graph");
  // dz/dy[i] = x[i]
  check(near(gy.data()[0], 1.0) && near(gy.data()[1], 2.0) && near(gy.data()[2], 3.0),
    "grad_y == x via create_graph");
}

// ══════════════════════════════════════════════════════════════════════════════
//  Bug 3 — reshape not differentiable (VJP seed had requires_grad=false)
//
//  f(x) = sum(reshape(x * x, {4})) with x shape [2,2]
//  df/dx[i] = 2*x[i]
//  d²f/dx[i]² = 2  (constant Hessian diagonal)
//
//  Before the fix: grad_tensor from create_graph had no computation graph
//  (because reshape's vjp_fn returned a dead Tensor), so calling grad() again
//  would throw "input not connected".
// ══════════════════════════════════════════════════════════════════════════════

static void test_reshape_differentiable()
{
  std::cout << "\n[Bug 3] reshape differentiable (create_graph seed)\n";

  Tensor x({1.0, 2.0, 3.0, 4.0}, {2, 2}, true);

  // f(x) = sum( reshape(x*x, {4}) )
  auto f = [](const Tensor& t) -> Tensor {
    return sum(mul(t, t).reshape({4}));
  };

  Tensor fx = f(x);
  Tensor gx = functional::grad(fx, x, false, /*create_graph=*/true);

  // First-order: grad == 2*x
  check(near(gx.data()[0], 2.0) && near(gx.data()[1], 4.0) &&
    near(gx.data()[2], 6.0) && near(gx.data()[3], 8.0),
    "first-order grad == 2*x");

  // Second-order: requires gx itself to have a graph → d(2x)/dx = 2
  bool second_ok = false;
  try {
    Tensor ggx = functional::grad(sum(gx), x, false, false);
    second_ok = near(ggx.data()[0], 2.0) && near(ggx.data()[1], 2.0) &&
        near(ggx.data()[2], 2.0) && near(ggx.data()[3], 2.0);
  } catch (...) {}

  check(second_ok, "second-order Hessian diag == 2 (reshape graph alive)");
}

// ══════════════════════════════════════════════════════════════════════════════
//  Bug 4 — matmul vjp created disconnected leaf tensors via reduce_broadcast
//
//  For the common non-broadcast case:  C = A @ B  with A[2×3], B[3×2]
//  dL/dA = dC @ Bᵀ  and  dL/dB = Aᵀ @ dC  must stay on the graph
//  so that second-order derivatives work.
//
//  f(A) = sum( A @ B ) → df/dA[i,j] = Σ_k B[j,k]  = row-sum of B
//  d²f/dA²  (w.r.t. any A element) = 0  because f is linear in A.
// ══════════════════════════════════════════════════════════════════════════════

static void test_matmul_vjp_graph()
{
  std::cout << "\n[Bug 4] matmul vjp no disconnected leaf nodes\n";

  // A [2×3], B [3×2] — non-batched, non-broadcast → dA_eff.shape == a_vjp.shape
  Tensor A({ 1.0, 2.0, 3.0,
     4.0, 5.0, 6.0 }, {2, 3}, true);
  Tensor B({ 1.0, 0.0,
     0.0, 1.0,
     1.0, 1.0 }, {3, 2}, true);

  Tensor C  = matmul(A, B);    // [2×2]
  Tensor sc = sum(C);

  // First-order gradients via create_graph
  Tensor gA = functional::grad(sc, A, false, /*create_graph=*/true);
  Tensor gB = functional::grad(sc, B, false, true);

  // df/dA[i,j] = Σ_k B[j,k] → B row-sums: {1,1}, {1,1}, {2,2}
  // gA shape [2×3]
  check(gA.numel() == 6, "gA numel == 6");
  bool ga_ok = near(gA.data()[0], 1.0) && near(gA.data()[1], 1.0) &&
       near(gA.data()[2], 2.0) && near(gA.data()[3], 1.0) &&
       near(gA.data()[4], 1.0) && near(gA.data()[5], 2.0);
  check(ga_ok, "gA values correct");

  // Second-order: f is linear in A → d(gA)/dA = 0
  bool second_ok = false;
  try {
    Tensor ggA = functional::grad(sum(gA), A, false, false);
    bool all_zero = true;
    for (double v : ggA.data()) if (!near(v, 0.0)) { all_zero = false; break; }
    second_ok = all_zero;
  } catch (...) {}

  check(second_ok, "second-order matmul grad == 0 (no disconnected leaf)");

  // df/dB[j,k] = Σ_i A[i,j] → B col-sums: A[:,0]={5}, A[:,1]={7}, A[:,2]={9}
  // gB shape [3×2]
  check(gB.numel() == 6, "gB numel == 6");
  bool gb_ok = near(gB.data()[0], 5.0) && near(gB.data()[1], 5.0) &&
       near(gB.data()[2], 7.0) && near(gB.data()[3], 7.0) &&
       near(gB.data()[4], 9.0) && near(gB.data()[5], 9.0);
  check(gb_ok, "gB values correct");
}

// ══════════════════════════════════════════════════════════════════════════════
//  Smoke test — standard (non-broadcast) mul backward still works
// ══════════════════════════════════════════════════════════════════════════════

static void test_mul_same_shape_regression()
{
  std::cout << "\n[Regression] mul same-shape backward\n";
  Tensor x({2.0, 3.0}, {2}, true);
  Tensor y({5.0, 4.0}, {2}, true);
  Tensor z = sum(mul(x, y)); // 10 + 12 = 22
  check(near(z.item(), 22.0), "forward == 22");
  z.backward();
  check(near(x.grad()[0], 5.0) && near(x.grad()[1], 4.0), "grad_x == y");
  check(near(y.grad()[0], 2.0) && near(y.grad()[1], 3.0), "grad_y == x");
}

static void test_mul_self_alias_gradient()
{
  std::cout << "\n[Regression] mul(x, x) alias gradient\n";
  Tensor x({3.0, -4.0}, {2}, true);
  Tensor z = sum(mul(x, x));
  z.backward();
  check(near(x.grad()[0], 6.0) && near(x.grad()[1], -8.0), "grad_x == 2*x");
}

static void test_matmul_backward_numerical()
{
  std::cout << "\n[Regression] matmul backward finite-difference\n";
  Tensor A({1.0, 2.0, 3.0, 4.0}, {2, 2}, true);
  Tensor B_fixed({0.5, -1.0, 1.5, 2.0}, {2, 2}, false);
  Tensor A_fixed({1.0, 2.0, 3.0, 4.0}, {2, 2}, false);
  Tensor B({0.5, -1.0, 1.5, 2.0}, {2, 2}, true);

  auto fA = [&B_fixed](const Tensor& A_in) { return sum(matmul(A_in, B_fixed)); };
  auto fB = [&A_fixed](const Tensor& B_in) { return sum(matmul(A_fixed, B_in)); };

  Tensor gA  = functional::grad(fA(A), A, false, false);
  Tensor gAn = functional::numerical_grad(fA, A, 1e-6);
  Tensor gB  = functional::grad(fB(B), B, false, false);
  Tensor gBn = functional::numerical_grad(fB, B, 1e-6);

  bool okA = true, okB = true;
  for (std::size_t i = 0; i < gA.numel(); ++i) okA = okA && near(gA.data()[i], gAn.data()[i], 1e-6);
  for (std::size_t i = 0; i < gB.numel(); ++i) okB = okB && near(gB.data()[i], gBn.data()[i], 1e-6);
  check(okA, "dA matches numerical grad");
  check(okB, "dB matches numerical grad");
}

static void test_gradcheck_suite()
{
  std::cout << "\n[Gradcheck] add/mul/matmul/sum/activation\n";
  Tensor x({0.2, -0.4, 0.7}, {3}, true);
  Tensor y({1.5, -2.0, 0.5}, {3}, false);

  gradcheck("add", [&y](const Tensor& t) { return sum(add(t, y)); }, x);
  gradcheck("mul", [&y](const Tensor& t) { return sum(mul(t, y)); }, x);
  gradcheck("sum", [](const Tensor& t) { return sum(t); }, x);
  gradcheck("tanh_act", [](const Tensor& t) { return sum(core::tanh_act(t)); }, x);
  gradcheck("sigmoid_act", [](const Tensor& t) { return sum(core::sigmoid_act(t)); }, x);

  Tensor A({1.0, -0.5, 0.2, 1.3}, {2, 2}, true);
  Tensor B({0.4, -1.2, 2.0, 0.7}, {2, 2}, false);
  gradcheck("matmul", [&B](const Tensor& T) { return sum(matmul(T, B)); }, A);
}

static void test_alias_reshape_backward()
{
  std::cout << "\n[View] alias + reshape + backward\n";
  Tensor x({1.0, -2.0, 3.0, -4.0}, {2, 2}, true);
  Tensor y = x.reshape({4});
  Tensor z = sum(mul(y, y));
  z.backward();
  check(near(x.grad()[0], 2.0) && near(x.grad()[1], -4.0) &&
        near(x.grad()[2], 6.0) && near(x.grad()[3], -8.0),
        "grad(x) == 2x through reshape view");
}

static void test_reshape_matmul_chain()
{
  std::cout << "\n[View] reshape -> matmul chain\n";
  Tensor x({1.0, 2.0, 3.0, 4.0}, {4}, true);
  Tensor A = x.reshape({2, 2});
  Tensor B({1.0, 0.0, 0.0, 1.0}, {2, 2}, false);
  Tensor loss = sum(matmul(A, B));
  loss.backward();
  check(near(x.grad()[0], 1.0) && near(x.grad()[1], 1.0) &&
        near(x.grad()[2], 1.0) && near(x.grad()[3], 1.0),
        "reshape->matmul backward is correct");
}

static void test_shared_storage_mutation()
{
  std::cout << "\n[View] shared storage mutation\n";
  Tensor x({1.0, 2.0, 3.0, 4.0}, {2, 2}, false);
  Tensor b = x.reshape({4});
  b.data()[0] = 999.0;
  check(near(x.data()[0], 999.0), "reshape view shares storage with source");
}

static std::size_t read_rss_kb()
{
  std::ifstream ifs("/proc/self/status");
  std::string line;
  while (std::getline(ifs, line)) {
    if (line.rfind("VmRSS:", 0) == 0) {
      std::string digits;
      for (char ch : line) if (std::isdigit(static_cast<unsigned char>(ch))) digits.push_back(ch);
      if (!digits.empty()) return static_cast<std::size_t>(std::stoull(digits));
    }
  }
  return 0;
}

static void test_backward_release_graph_stress()
{
  std::cout << "\n[Stress] backward release_graph memory stability\n";
  int iters = 10000;
  if (const char* env_iters = std::getenv("UML_AUTOGRAD_STRESS_ITERS")) {
    const int parsed = std::atoi(env_iters);
    if (parsed > 0) iters = parsed;
  }
  const std::size_t rss_before = read_rss_kb();
  for (int iter = 0; iter < iters; ++iter) {
    Tensor a(std::vector<double>(64 * 64, 0.5), {64, 64}, true);
    Tensor b(std::vector<double>(64 * 64, 1.5), {64, 64}, true);
    Tensor loss = mean(matmul(a, b));
    loss.backward(/*retain_graph=*/false, /*create_graph=*/false);
  }
  const std::size_t rss_after = read_rss_kb();
  const std::size_t rss_growth = (rss_after >= rss_before) ? (rss_after - rss_before) : 0;
  // Heuristic guard: transient allocator noise is allowed, monotonic growth is not.
  // Under ASan/MSan each allocation carries shadow + redzone overhead and the
  // allocator does not return RSS to the OS — raise the limit accordingly.
#if defined(__SANITIZE_ADDRESS__) || defined(__SANITIZE_MEMORY__)
  constexpr std::size_t rss_limit_kb = 64 * 1024; // 64 MB under sanitizers
#else
  constexpr std::size_t rss_limit_kb = 12 * 1024; // 12 MB in normal builds
#endif
  check(rss_growth < rss_limit_kb,
        "RSS growth < " + std::to_string(rss_limit_kb / 1024) +
        "MB over " + std::to_string(iters) + " backward passes");
}

// ══════════════════════════════════════════════════════════════════════════════
//  main
// ══════════════════════════════════════════════════════════════════════════════

int main()
{
  std::cout << "=== autograd regression tests ===\n";

  test_mul_broadcast();
  test_vjp_cast_safety();
  test_reshape_differentiable();
  test_matmul_vjp_graph();
  test_mul_same_shape_regression();
  test_mul_self_alias_gradient();
  test_matmul_backward_numerical();
  test_gradcheck_suite();
  test_alias_reshape_backward();
  test_reshape_matmul_chain();
  test_shared_storage_mutation();
  test_backward_release_graph_stress();

  std::cout << "\n      \n";
  std::cout << "PASSED: " << g_pass << "  FAILED: " << g_fail << "\n";
  return (g_fail == 0) ? 0 : 1;
}