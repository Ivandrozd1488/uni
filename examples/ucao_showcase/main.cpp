/**
 * @file main.cpp
 * @brief Internal UCAO subsystem showcase for unified_ml.
 *
 * This executable demonstrates that UCAO is integrated as a coherent internal
 * system inside the library rather than as a loose collection of tests.
 */

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>

#include <models/deep_onet/model.hpp>
#include <models/pideeponet/pideeponet.hpp>
#include <models/sindy/sindy.hpp>
#include <models/transformer/transformer_block.hpp>
#include <ucao/system.hpp>

namespace {

using Clock = std::chrono::steady_clock;

void print_registry() {
    std::cout << "internal engine registry\n";
    for (const auto& entry : ucao::engine::kRegistry) {
        const auto selection = ucao::engine::select_runtime(entry.family);
        std::cout << "  - " << entry.family_name << " -> " << entry.descriptor.name
                  << " [registry=" << (entry.descriptor.enabled ? "enabled" : "disabled")
                  << ", runtime=" << (selection.selected ? "enabled" : "disabled")
                  << ", preference=" << ucao::engine::to_string(selection.preference) << "]\n";
    }
    std::cout << '\n';
}

void print_header() {
    std::cout << "unified_ml internal subsystem showcase\n";
    std::cout << "subsystem: UCAO\n";
    std::cout << "version:   " << ucao::system::version_string << "\n\n";
    print_registry();
}

bool showcase_field_layer() {
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

    std::cout << "[field-layer]\n";
    std::cout << "  samples:            " << sample_count << '\n';
    std::cout << "  elapsed_us:         " << elapsed_us << '\n';
    std::cout << "  samples_per_second: " << std::fixed << std::setprecision(2) << samples_per_second << '\n';
    std::cout << "  mean_norm_error:    " << std::scientific << mean_norm_error << '\n';
    std::cout << "  checksum:           " << std::fixed << std::setprecision(6) << checksum << "\n\n";

    return mean_norm_error < 1e-3;
}

bool showcase_rotor_chain() {
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

    std::cout << "[rotor-chain]\n";
    std::cout << "  steps:              2000\n";
    std::cout << "  elapsed_us:         " << elapsed_us << '\n';
    std::cout << "  max_rotor_error:    " << std::scientific << max_error << "\n\n";

    return max_error < 1e-4f;
}

} // namespace

int main() {
    static_assert(deep_onet::DeepONet::uses_ucao_engine());
    static_assert(pideeponet::PIDeepONet::uses_ucao_engine());
    static_assert(transformer::TransformerBlock::uses_ucao_engine());
    static_assert(transformer::TransformerEncoder::uses_ucao_engine());
    static_assert(sindy::SINDy::uses_ucao_engine());

    auto policy = ucao::engine::default_policy();
    policy.ucao_globally_enabled = true;
    policy.family_policies[static_cast<std::size_t>(ucao::engine::ModelFamily::Transformer) - 1].preference =
        ucao::engine::EnginePreference::PreferDisabled;
    ucao::engine::set_policy(policy);

    print_header();
    const bool field_ok = showcase_field_layer();
    const bool rotor_ok = showcase_rotor_chain();

    const auto deep_selection = deep_onet::DeepONet::runtime_selection();
    const auto pideeponet_selection = pideeponet::PIDeepONet::runtime_selection();
    const auto transformer_selection = transformer::TransformerBlock::runtime_selection();
    const auto sindy_selection = sindy::SINDy::runtime_selection();

    const bool registry_ok =
        deep_selection.descriptor.kind == ucao::engine::EngineKind::DeepOperatorGeometry && deep_selection.selected &&
        pideeponet_selection.descriptor.kind == ucao::engine::EngineKind::PhysicsInformedOperator && pideeponet_selection.selected &&
        transformer_selection.descriptor.kind == ucao::engine::EngineKind::SequenceGeometry && !transformer_selection.selected &&
        sindy_selection.descriptor.kind == ucao::engine::EngineKind::SparseGeometryDynamics && sindy_selection.selected;

    std::cout << "policy: transformer runtime path is intentionally disabled for showcase\n";
    std::cout << "registry: " << (registry_ok ? "PASSED" : "FAILED") << '\n';
    std::cout << "overall: " << ((field_ok && rotor_ok && registry_ok) ? "PASSED" : "FAILED") << '\n';
    return (field_ok && rotor_ok && registry_ok) ? 0 : 1;
}
