// test_core_activations.cpp
// Unit tests for src/core/activations.cpp
// Previously 50% — smoke test only hit relu. This covers:
//   exp_act, tanh_act, sigmoid_act, gelu_act, relu_act
// For each: forward values, gradient (backward), batch tensor.

#include <autograd/autograd.h>
#include <core/activations.hpp>

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace {

int g_pass = 0, g_fail = 0;

void check(bool ok, const std::string& name) {
    if (ok) { std::cout << "PASS  " << name << "\n"; ++g_pass; }
    else    { std::cout << "FAIL  " << name << "\n"; ++g_fail; }
}

bool near(double a, double b, double tol = 1e-6) { return std::fabs(a-b) < tol; }

autograd::Tensor scalar_leaf(double v) {
    return autograd::Tensor({v}, {1}, true);
}

double val(const autograd::Tensor& t) { return t.data()[0]; }

// ── exp_act ──────────────────────────────────────────────────────────────────
void test_exp_act() {
    auto x = scalar_leaf(1.0);
    auto y = core::exp_act(x);
    check(near(val(y), std::exp(1.0)), "exp_act forward value");

    y.backward();
    // d/dx exp(x) = exp(x) → at x=1: exp(1)
    check(near(x.grad()[0], std::exp(1.0)), "exp_act backward grad");

    // negative input
    auto x2 = scalar_leaf(-2.0);
    auto y2 = core::exp_act(x2);
    check(near(val(y2), std::exp(-2.0)), "exp_act negative input");

    // batch
    auto xb = autograd::Tensor({0.0, 1.0, 2.0}, {3}, true);
    auto yb = core::exp_act(xb);
    check(near(yb.data()[2], std::exp(2.0)), "exp_act batch last element");
}

// ── tanh_act ─────────────────────────────────────────────────────────────────
void test_tanh_act() {
    auto x = scalar_leaf(0.5);
    auto y = core::tanh_act(x);
    check(near(val(y), std::tanh(0.5)), "tanh_act forward value");

    y.backward();
    // d/dx tanh(x) = 1 - tanh^2(x)
    double expected_grad = 1.0 - std::tanh(0.5) * std::tanh(0.5);
    check(near(x.grad()[0], expected_grad), "tanh_act backward grad");

    // tanh(0) == 0
    auto x0 = scalar_leaf(0.0);
    check(near(val(core::tanh_act(x0)), 0.0), "tanh_act at 0");

    // batch
    auto xb = autograd::Tensor({-1.0, 0.0, 1.0}, {3}, true);
    auto yb = core::tanh_act(xb);
    check(near(yb.data()[0], std::tanh(-1.0)), "tanh_act batch negative");
    check(near(yb.data()[2], std::tanh(1.0)),  "tanh_act batch positive");
}

// ── sigmoid_act ──────────────────────────────────────────────────────────────
void test_sigmoid_act() {
    auto sigmoid = [](double z){ return 1.0 / (1.0 + std::exp(-z)); };

    auto x = scalar_leaf(0.0);
    auto y = core::sigmoid_act(x);
    check(near(val(y), 0.5), "sigmoid_act at 0 == 0.5");

    y.backward();
    // d/dx sigmoid(0) = 0.25
    check(near(x.grad()[0], 0.25), "sigmoid_act backward at 0");

    auto x2 = scalar_leaf(2.0);
    auto y2 = core::sigmoid_act(x2);
    check(near(val(y2), sigmoid(2.0)), "sigmoid_act positive input");

    // batch
    auto xb = autograd::Tensor({-3.0, 0.0, 3.0}, {3}, true);
    auto yb = core::sigmoid_act(xb);
    check(near(yb.data()[1], 0.5), "sigmoid_act batch middle");
    check(yb.data()[2] > 0.9,      "sigmoid_act batch large positive > 0.9");
}

// ── gelu_act ─────────────────────────────────────────────────────────────────
void test_gelu_act() {
    // GELU(0) = 0
    auto x0 = scalar_leaf(0.0);
    auto y0 = core::gelu_act(x0);
    check(near(val(y0), 0.0, 1e-5), "gelu_act at 0 == 0");

    // GELU(x) > 0 for large positive x (approximately x)
    auto xbig = scalar_leaf(5.0);
    auto ybig = core::gelu_act(xbig);
    check(val(ybig) > 4.0, "gelu_act large positive x ≈ x");

    // GELU(x) ≈ 0 for very negative x
    auto xneg = scalar_leaf(-5.0);
    auto yneg = core::gelu_act(xneg);
    check(std::fabs(val(yneg)) < 0.01, "gelu_act large negative ≈ 0");

    // backward doesn't crash
    auto x = scalar_leaf(1.0);
    auto y = core::gelu_act(x);
    y.backward();
    check(x.grad().size() == 1, "gelu_act backward ran without crash");

    // batch
    auto xb = autograd::Tensor({-2.0, 0.0, 2.0}, {3}, true);
    auto yb = core::gelu_act(xb);
    check(yb.data()[1] == 0.0 || std::fabs(yb.data()[1]) < 1e-5,
          "gelu_act batch at 0");
}

// ── relu_act ─────────────────────────────────────────────────────────────────
void test_relu_act() {
    auto xp = scalar_leaf(3.0);
    auto yp = core::relu_act(xp);
    check(near(val(yp), 3.0), "relu_act positive passthrough");

    auto xn = scalar_leaf(-2.0);
    auto yn = core::relu_act(xn);
    check(near(val(yn), 0.0), "relu_act negative clamped to 0");

    // gradient
    xp.zero_grad();
    yp.backward();
    check(near(xp.grad()[0], 1.0), "relu_act grad positive == 1");

    // batch
    auto xb = autograd::Tensor({-1.0, 0.5, 2.0}, {3}, true);
    auto yb = core::relu_act(xb);
    check(near(yb.data()[0], 0.0), "relu_act batch negative");
    check(near(yb.data()[2], 2.0), "relu_act batch positive");
}

} // namespace

int main() {
    test_exp_act();
    test_tanh_act();
    test_sigmoid_act();
    test_gelu_act();
    test_relu_act();

    std::cout << "\n" << g_pass << " passed, " << g_fail << " failed\n";
    return g_fail ? 1 : 0;
}
