#include <cmath>
#include <iostream>

#include "ucao/ucao.hpp"

namespace {

bool test_field_layer_forward() {
    ucao::pinn::CliffordFieldLayer<3,3,0> layer(42);
    float coord[3] = {1.0f, 0.5f, -0.3f};
    float out[8]{};
    layer.forward(coord, out);
    float n2 = 0.0f;
    for (int i = 0; i < 8; ++i) {
        const float s = static_cast<float>(ucao::kernel::detail::reversion_sign<3>(i));
        n2 += s * out[i] * out[i];
    }
    return std::fabs(std::fabs(n2) - 1.0f) < 1e-3f;
}

bool test_field_layer_dual() {
    ucao::pinn::CliffordFieldLayer<3,3,0> layer(42);
    float x[3] = {0.2f, -0.1f, 0.4f};
    const auto dual = layer.forward_dual(ucao::kernel::DualMV<3,3,0>::embed_coord(x, 0));
    float base[8]{};
    float plus[8]{};
    float minus[8]{};
    layer.forward(x, base);
    x[0] += 1e-3f;
    layer.forward(x, plus);
    x[0] -= 2e-3f;
    layer.forward(x, minus);
    for (int i = 0; i < 8; ++i) {
        const float fd = (plus[i] - minus[i]) / (2e-3f);
        if (std::fabs(fd - dual.dual[i]) > 3e-2f) return false;
    }
    return true;
}

bool test_pinn_loss() {
    ucao::pinn::CliffordFieldLayer<3,3,0> layer(42);
    for (auto& row : layer.weights) for (float& v : row) v = 0.0f;
    for (auto& row : layer.bias) for (float& v : row) v = 0.0f;
    layer.bias[0][0] = 1.0f;
    float pts[3] = {0.5f, 0.5f, 0.5f};
    float J[8]{};
    const float loss = ucao::pinn::clifford_pinn_loss_exact(layer, pts, J, 1);
    return loss >= 0.0f && std::fabs(loss) < 1e-6f;
}

bool test_clifford_layer_graph() {
    ucao::ml::CliffordLayer<3,3,0> layer;
    autograd::Tensor input(std::vector<double>{1,0,0,0,0,0,0,0}, std::vector<std::size_t>{8}, true);
    try {
        auto out = layer.forward(input);
        auto out_sum = autograd::sum(out);
        out_sum.backward();
        return true;
    } catch (...) {
        return false;
    }
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
    run("test_field_layer_forward", test_field_layer_forward);
    run("test_field_layer_dual", test_field_layer_dual);
    run("test_pinn_loss", test_pinn_loss);
    run("test_clifford_layer_graph", test_clifford_layer_graph);
    std::cout << "PASSED: " << passed << " FAILED: " << failed << '\n';
    return failed == 0 ? 0 : 1;
}
