#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

#include "ucao/ucao.hpp"

namespace {

using Clock = std::chrono::steady_clock;

bool run_field_layer_demo() {
    ucao::pinn::CliffordFieldLayer<3, 3, 0> layer(42);

    constexpr int sample_count = 4096;
    float coord[3]{};
    float out[8]{};

    double norm_error_sum = 0.0;
    double checksum = 0.0;

    const auto start = Clock::now();
    for (int i = 0; i < sample_count; ++i) {
        coord[0] = std::sin(0.001f * static_cast<float>(i));
        coord[1] = std::cos(0.002f * static_cast<float>(i));
        coord[2] = 0.25f * std::sin(0.003f * static_cast<float>(i));
        layer.forward(coord, out);

        float n2 = 0.0f;
        for (int j = 0; j < 8; ++j) {
            const float s = static_cast<float>(ucao::kernel::detail::reversion_sign<3>(j));
            n2 += s * out[j] * out[j];
            checksum += out[j];
        }
        norm_error_sum += std::abs(std::abs(static_cast<double>(n2)) - 1.0);
    }
    const auto end = Clock::now();

    const auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    const double mean_norm_error = norm_error_sum / static_cast<double>(sample_count);
    const double samples_per_second = (sample_count * 1e6) / static_cast<double>(elapsed_us > 0 ? elapsed_us : 1);

    std::cout << "UCAO field-layer demo\n";
    std::cout << "  samples:            " << sample_count << '\n';
    std::cout << "  elapsed_us:         " << elapsed_us << '\n';
    std::cout << "  samples_per_second: " << std::fixed << std::setprecision(2) << samples_per_second << '\n';
    std::cout << "  mean_norm_error:    " << std::scientific << mean_norm_error << '\n';
    std::cout << "  checksum:           " << std::fixed << std::setprecision(6) << checksum << '\n';

    return mean_norm_error < 1e-3;
}

bool run_rotor_chain_demo() {
    ucao::combat::FPRotorChain<3, 3, 0, 4, 16> chain;
    ucao::kernel::FPMultivector<3, 3, 0> increment;
    increment.comps[0] = ucao::kernel::FP_SCALE;
    increment.comps[3] = ucao::kernel::FP_SCALE / 8192;
    increment.reorthogonalize(3);

    const auto start = Clock::now();
    for (int step = 0; step < 2000; ++step) {
        chain.update(0, increment, step);
    }
    const auto end = Clock::now();

    const auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    const float max_error = chain.max_rotor_error();

    std::cout << "UCAO rotor-chain demo\n";
    std::cout << "  steps:              2000\n";
    std::cout << "  elapsed_us:         " << elapsed_us << '\n';
    std::cout << "  max_rotor_error:    " << std::scientific << max_error << '\n';

    return max_error < 1e-4f;
}

} // namespace

int main() {
    const bool field_ok = run_field_layer_demo();
    const bool rotor_ok = run_rotor_chain_demo();

    std::cout << "UCAO demo result: " << ((field_ok && rotor_ok) ? "PASSED" : "FAILED") << '\n';
    return (field_ok && rotor_ok) ? 0 : 1;
}
