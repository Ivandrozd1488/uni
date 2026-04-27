#include <cmath>
#include <iostream>

#include "ucao/ucao.hpp"

namespace {

bool same_mv(const ucao::kernel::FPMultivector<3,3,0>& a, const ucao::kernel::FPMultivector<3,3,0>& b) {
    for (int i = 0; i < 8; ++i) {
        if (a.comps[i] != b.comps[i]) return false;
    }
    return true;
}

bool test_motor_apply_identity() {
    ucao::kernel::FPMultivector<3,3,0> M;
    ucao::kernel::FPMultivector<3,3,0> X;
    M.comps[0] = ucao::kernel::FP_SCALE;
    X.comps[1] = ucao::kernel::FP_SCALE / 2;
    X.comps[3] = ucao::kernel::FP_SCALE / 4;
    const auto result = ucao::combat::MotorKinematics<3,3,0>::apply(M, X);
    return same_mv(result, X);
}

bool test_motor_compose() {
    ucao::kernel::FPMultivector<3,3,0> M1;
    M1.comps[0] = ucao::kernel::FP_SCALE;
    M1.comps[3] = ucao::kernel::FP_SCALE / 8;
    M1.reorthogonalize(3);
    const auto M12 = ucao::combat::MotorKinematics<3,3,0>::compose(M1, M1);
    const auto expected = ucao::kernel::fp_gp<3,3,0>(M1, M1);
    for (int i = 0; i < 8; ++i) {
        if (std::llabs(M12.comps[i] - expected.comps[i]) > 4) return false;
    }
    return true;
}

bool test_lerp_normalized() {
    ucao::kernel::FPMultivector<3,3,0> M1;
    ucao::kernel::FPMultivector<3,3,0> M2;
    M1.comps[0] = ucao::kernel::FP_SCALE;
    M2.comps[0] = ucao::kernel::FP_SCALE;
    M2.comps[3] = ucao::kernel::FP_SCALE / 4;
    M2.reorthogonalize(3);
    const auto half = ucao::combat::MotorKinematics<3,3,0>::lerp_normalized(M1, M2, ucao::kernel::FP_SCALE / 2);
    return half.is_unit_rotor(ucao::kernel::FP_SCALE >> 10);
}

bool test_rotor_chain_identity() {
    ucao::combat::FPRotorChain<3,3,0,4> chain;
    ucao::kernel::FPMultivector<3,3,0> X;
    X.comps[5] = ucao::kernel::FP_SCALE / 3;
    const auto result = chain.apply_chain(X);
    return same_mv(result, X);
}

bool test_rotor_drift_bound() {
    ucao::combat::FPRotorChain<3,3,0,4,10> chain;
    ucao::kernel::FPMultivector<3,3,0> increment;
    increment.comps[0] = ucao::kernel::FP_SCALE;
    increment.comps[3] = ucao::kernel::FP_SCALE / 8192;
    increment.reorthogonalize(3);
    for (int step = 0; step < 1000; ++step) {
        chain.update(0, increment, step);
    }
    return chain.max_rotor_error() < 1e-5f;
}

}

int main() {
    int passed = 0;
    int failed = 0;
    auto run = [&](const char* name, bool (*fn)()) {
        const bool ok = fn();
        std::cout << name << ": " << (ok ? "PASSED" : "FAILED") << '\n';
        ok ? ++passed : ++failed;
    };
    run("test_motor_apply_identity", test_motor_apply_identity);
    run("test_motor_compose", test_motor_compose);
    run("test_lerp_normalized", test_lerp_normalized);
    run("test_rotor_chain_identity", test_rotor_chain_identity);
    run("test_rotor_drift_bound", test_rotor_drift_bound);
    std::cout << "PASSED: " << passed << " FAILED: " << failed << '\n';
    return failed == 0 ? 0 : 1;
}
