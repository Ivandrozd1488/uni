// test_core_optimizers.cpp
// Unit tests for src/core/optimizers.cpp
// Previously had 12% coverage.  Exercises all four optimizer classes:
//   SGD (with and without momentum), Adam, RMSProp, NAdam
// Checks that a parameter with a known gradient moves in the right direction,
// that zero_grad() clears gradients, that state save/load round-trips, and that
// clip_grad_norm() clamps large gradients.

#include <autograd/autograd.h>
#include <core/optimizers.hpp>

#include <cassert>
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

// Helper: make a scalar leaf tensor and hand-set a gradient of `grad_val`.
// Returns the tensor and manually sets node->grad = {grad_val}.
autograd::Tensor make_param(double val, double grad_val) {
    autograd::Tensor p({val}, {1}, true);
    p.backward(autograd::Tensor({1.0}, {1}));
    // reset data to val and grad to grad_val via a fresh scalar computation
    // Simpler: just build the tensor and call backward on a scaled version.
    // We'll use a simple: loss = p * grad_val  → grad of p is grad_val.
    (void)grad_val; // see below
    return p;
}

// Build a simple computation loss = p * scale, then backward so grad == scale.
autograd::Tensor make_leaf(double value) {
    return autograd::Tensor({value}, {1}, true);
}

double get_value(const autograd::Tensor& t) { return t.data()[0]; }

// ── SGD ──────────────────────────────────────────────────────────────────────
void test_sgd_basic() {
    // loss = p^2, grad = 2*p  → SGD step: p = p - lr*2*p
    auto p = make_leaf(3.0);
    auto loss = autograd::mul(p, p);           // p*p
    loss.backward();

    double grad = p.grad()[0];                  // should be 2*3 = 6
    check(std::fabs(grad - 6.0) < 1e-9, "SGD: grad of p^2 is 2*p");

    double lr = 0.1;
    double p_old = get_value(p);
    core::SGD optim({&p}, lr);
    optim.step();
    double p_new = get_value(p);

    check(p_new < p_old, "SGD: parameter moves toward minimum");
    check(std::fabs(p_new - (p_old - lr * grad)) < 1e-9,
          "SGD: exact step size p - lr*grad");
}

void test_sgd_zero_grad() {
    auto p = make_leaf(1.0);
    auto loss = autograd::mul(p, p);
    loss.backward();

    core::SGD optim({&p}, 0.01);
    optim.zero_grad();
    check(std::all_of(p.grad().begin(), p.grad().end(),
                      [](double g){ return g == 0.0; }),
          "SGD::zero_grad clears gradients");
}

void test_sgd_momentum() {
    // Two steps with momentum — parameter should still descend.
    auto p = make_leaf(2.0);
    core::SGD optim({&p}, 0.1, 0.9 /*momentum*/);

    for (int i = 0; i < 3; ++i) {
        optim.zero_grad();
        auto loss = autograd::mul(p, p);
        loss.backward();
        optim.step();
    }
    check(get_value(p) < 2.0, "SGD momentum: descends over 3 steps");
}

void test_sgd_weight_decay() {
    auto p = make_leaf(1.0);
    auto loss = autograd::mul(p, p);
    loss.backward();
    core::SGD optim({&p}, 0.1, 0.0, 0.0, 0.01 /*weight_decay*/);
    double before = get_value(p);
    optim.step();
    check(get_value(p) < before, "SGD weight_decay: param still decreases");
}

void test_sgd_clip_grad_norm() {
    auto p = make_leaf(1.0);
    // Artificially set a large gradient via a scaled loss
    auto large = autograd::mul(p, autograd::Tensor({100.0}, {1}));
    large.backward();
    double orig_grad = std::fabs(p.grad()[0]);
    check(orig_grad > 1.0, "SGD clip_grad: grad is large before clip");

    core::SGD optim({&p}, 0.01);
    optim.clip_grad_norm(1.0);
    double clipped = std::fabs(p.grad()[0]);
    check(clipped <= 1.0 + 1e-9, "SGD::clip_grad_norm <= max_norm");
}

void test_sgd_state_save_load() {
    auto p = make_leaf(1.0);
    auto loss = autograd::mul(p, p);
    loss.backward();
    core::SGD optim({&p}, 0.05, 0.9);
    optim.step();

    auto state = optim.save_state();
    check(state.lr == 0.05 && state.momentum == 0.9, "SGD save_state lr/momentum");

    auto p2 = make_leaf(1.0);
    auto loss2 = autograd::mul(p2, p2);
    loss2.backward();
    core::SGD optim2({&p2}, 0.01);
    optim2.load_state(state);
    auto state2 = optim2.save_state();
    check(state2.lr == state.lr, "SGD load_state round-trip lr");
}

// ── Adam ─────────────────────────────────────────────────────────────────────
void test_adam_descent() {
    auto p = make_leaf(3.0);
    core::Adam optim({&p}, 0.1);

    for (int i = 0; i < 5; ++i) {
        optim.zero_grad();
        auto loss = autograd::mul(p, p);
        loss.backward();
        optim.step();
    }
    check(get_value(p) < 3.0, "Adam: descends toward 0 over 5 steps");
}

void test_adam_zero_grad() {
    auto p = make_leaf(1.0);
    auto loss = autograd::mul(p, p);
    loss.backward();
    core::Adam optim({&p}, 0.01);
    optim.zero_grad();
    check(std::all_of(p.grad().begin(), p.grad().end(),
                      [](double g){ return g == 0.0; }),
          "Adam::zero_grad");
}

void test_adam_clip_grad_norm() {
    auto p = make_leaf(1.0);
    auto large = autograd::mul(p, autograd::Tensor({500.0}, {1}));
    large.backward();
    core::Adam optim({&p}, 0.01);
    optim.clip_grad_norm(1.0);
    check(std::fabs(p.grad()[0]) <= 1.0 + 1e-9, "Adam::clip_grad_norm");
}

void test_adam_state_save_load() {
    auto p = make_leaf(2.0);
    core::Adam optim({&p}, 0.001, 0.9, 0.999, 1e-8);

    for (int i = 0; i < 3; ++i) {
        optim.zero_grad();
        auto loss = autograd::mul(p, p);
        loss.backward();
        optim.step();
    }

    auto state = optim.save_state();
    check(state.lr == 0.001 && state.timestep == 3, "Adam save_state lr+timestep");

    auto p2 = make_leaf(2.0);
    core::Adam optim2({&p2}, 0.5);
    optim2.load_state(state);
    auto state2 = optim2.save_state();
    check(state2.lr == state.lr && state2.timestep == state.timestep,
          "Adam load_state round-trip");
}

// ── RMSProp ──────────────────────────────────────────────────────────────────
void test_rmsprop_descent() {
    auto p = make_leaf(3.0);
    core::RMSProp optim({&p}, 0.01);

    for (int i = 0; i < 5; ++i) {
        optim.zero_grad();
        auto loss = autograd::mul(p, p);
        loss.backward();
        optim.step();
    }
    check(get_value(p) < 3.0, "RMSProp: descends");
}

void test_rmsprop_state() {
    auto p = make_leaf(1.0);
    core::RMSProp optim({&p}, 0.01);
    auto state = optim.save_state();
    check(state.lr == 0.01, "RMSProp save_state lr");

    auto p2 = make_leaf(1.0);
    core::RMSProp optim2({&p2}, 0.1);
    optim2.load_state(state);
    check(optim2.save_state().lr == 0.01, "RMSProp load_state");
}

// ── NAdam ────────────────────────────────────────────────────────────────────
void test_nadam_descent() {
    auto p = make_leaf(4.0);
    core::NAdam optim({&p}, 0.1);

    for (int i = 0; i < 5; ++i) {
        optim.zero_grad();
        auto loss = autograd::mul(p, p);
        loss.backward();
        optim.step();
    }
    check(get_value(p) < 4.0, "NAdam: descends");
}

void test_nadam_state() {
    auto p = make_leaf(1.0);
    core::NAdam optim({&p}, 0.002);
    auto st = optim.save_state();
    check(st.lr == 0.002, "NAdam save_state lr");

    auto p2 = make_leaf(1.0);
    core::NAdam optim2({&p2}, 0.5);
    optim2.load_state(st);
    check(optim2.save_state().lr == 0.002, "NAdam load_state");
}

} // namespace

int main() {
    test_sgd_basic();
    test_sgd_zero_grad();
    test_sgd_momentum();
    test_sgd_weight_decay();
    test_sgd_clip_grad_norm();
    test_sgd_state_save_load();

    test_adam_descent();
    test_adam_zero_grad();
    test_adam_clip_grad_norm();
    test_adam_state_save_load();

    test_rmsprop_descent();
    test_rmsprop_state();

    test_nadam_descent();
    test_nadam_state();

    std::cout << "\n" << g_pass << " passed, " << g_fail << " failed\n";
    return g_fail ? 1 : 0;
}
