#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <new>
#include <stdexcept>

#include "ucao/ucao.hpp"

std::size_t g_allocs = 0;
void* operator new(std::size_t sz) {
    ++g_allocs;
    if (void* p = std::malloc(sz)) return p;
    throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }

namespace {

bool test_sign_table_cl30() {
    using namespace ucao::kernel;
    return kGPTable<3,3,0>.sign(0b001, 0b001) == +1 &&
           kGPTable<3,3,0>.sign(0b001, 0b010) == +1 &&
           kGPTable<3,3,0>.sign(0b010, 0b001) == -1 &&
           kGPTable<4,3,1>.sign(0b0001, 0b0001) == +1 &&
           kGPTable<4,3,1>.sign(0b1000, 0b1000) == -1 &&
           kGPTable<3,3,0>.nnz_[0] == 8;
}

bool test_fp_mul() {
    using namespace ucao::kernel;
    return fp_mul(FP_SCALE, FP_SCALE) == FP_SCALE && fp_mul(FP_SCALE / 2, FP_SCALE * 2) == FP_SCALE;
}

bool test_fp_inv_sqrt() {
    using namespace ucao::kernel;
    const fp64 x = FP_SCALE * 4;
    const fp64 result = fp_inv_sqrt_nr(x, 3);
    return std::llabs(result - FP_SCALE / 2) < (FP_SCALE >> 25);
}

bool test_fp_gp_cl30() {
    using namespace ucao::kernel;
    FPMultivector<3,3,0> e1;
    FPMultivector<3,3,0> e2;
    e1.comps[0b001] = FP_SCALE;
    e2.comps[0b010] = FP_SCALE;
    auto e12 = fp_gp<3,3,0>(e1, e2);
    if (e12.comps[0b011] != FP_SCALE) return false;
    for (int i = 0; i < 8; ++i) {
        if (i != 0b011 && e12.comps[i] != 0) return false;
    }
    return true;
}

bool test_reorthogonalize() {
    using namespace ucao::kernel;
    FPMultivector<3,3,0> r;
    r.comps[0] = FP_SCALE;
    r.reorthogonalize(3);
    r.comps[0] += FP_SCALE >> 3;
    r.stabilize_rotor();
    return r.rotor_error() < 1e-5f;
}

bool test_avx512_shuffle_matches_scalar() {
    using namespace ucao::kernel;
    alignas(64) float a[16]{};
    alignas(64) float b[16]{};
    alignas(64) float ref8[16]{};
    alignas(64) float simd8[16]{};
    alignas(64) float ref16[16]{};
    alignas(64) float simd16[16]{};
    for (int i = 0; i < 16; ++i) {
        a[i] = 0.1f * static_cast<float>(i + 1);
        b[i] = -0.05f * static_cast<float>(i + 3);
    }
    gp_single<3,3,0>(ref8, a, b);
#ifdef __AVX512F__
    ucao::kernel::detail::gp_single_avx512_shuffled<3,3,0>(simd8, a, b);
    for (int i = 0; i < 8; ++i) if (std::fabs(ref8[i] - simd8[i]) > 1e-5f) return false;
    gp_single<4,3,1>(ref16, a, b);
    ucao::kernel::detail::gp_single_avx512_shuffled<4,3,1>(simd16, a, b);
    for (int i = 0; i < 16; ++i) if (std::fabs(ref16[i] - simd16[i]) > 1e-5f) return false;
#endif
    return true;
}

bool test_dual_mv_gp() {
    using namespace ucao::kernel;
    DualMV<3,3,0> a;
    DualMV<3,3,0> b;
    a.real[0b001] = 1.0f;
    a.dual[0b001] = 1.0f;
    b.real[0b001] = 1.0f;
    auto c = gp_d(a, b);
    return std::fabs(c.real[0] - 1.0f) < 1e-6f && std::fabs(c.dual[0] - 1.0f) < 1e-6f;
}

struct QuadraticLayer {
    ucao::kernel::DualMV<2,2,0> forward_dual(const ucao::kernel::DualMV<2,2,0>& x) const noexcept {
        ucao::kernel::DualMV<2,2,0> out;
        const float x0 = x.real[0b01];
        const float x1 = x.real[0b10];
        const float dx0 = x.dual[0b01];
        const float dx1 = x.dual[0b10];
        out.real[0] = x0 * x0 + x1 * x1;
        out.dual[0] = 2.0f * x0 * dx0 + 2.0f * x1 * dx1;
        return out;
    }
};

bool test_vector_derivative() {
    const float x[2] = {1.5f, -0.25f};
    const auto grad = ucao::kernel::VectorDerivative<2,2,0>::template apply<QuadraticLayer>(QuadraticLayer{}, x);
    return std::fabs(grad[0b01] - 3.0f) < 1e-4f && std::fabs(grad[0b10] + 0.5f) < 1e-4f;
}

bool test_mv_tensor_layout() {
    using Tensor = ucao::kernel::MVTensor<8>;
    Tensor t(8, Tensor::Layout::AoS);
    for (int b = 0; b < 8; ++b) for (int d = 0; d < 8; ++d) t.at(b,d) = static_cast<float>(b * 10 + d);
    t.transpose_to(Tensor::Layout::SoA);
    if (t.at(3,5) != 35.0f) return false;
    t.transpose_to(Tensor::Layout::AoS);
    if (t.at(6,7) != 67.0f) return false;
    Tensor A(8, Tensor::Layout::AoS);
    Tensor B(8, Tensor::Layout::AoS);
    Tensor C(8, Tensor::Layout::AoS);
    for (int b = 0; b < 8; ++b) {
        A.at(b,1) = 1.0f;
        B.at(b,2) = 1.0f;
    }
    ucao::kernel::gp_dispatch_layout<3,3,0>(C, A, B);
    return C.layout() == Tensor::Layout::AoS && C.at(0,3) == 1.0f;
}

bool test_mv_tensor_checked_contracts() {
    using Tensor = ucao::kernel::MVTensor<8>;
    Tensor soa_tensor(4, Tensor::Layout::SoA);
    Tensor aos_tensor(4, Tensor::Layout::AoS);
    bool threw_index = false;
    bool threw_layout = false;
    bool threw_batch = false;

    try {
        (void)soa_tensor.at_checked(6, 1);
    } catch (const std::out_of_range&) {
        threw_index = true;
    }

    try {
        (void)aos_tensor.soa_row_checked(0);
    } catch (const std::invalid_argument&) {
        threw_layout = true;
    }

    try {
        Tensor A(4, Tensor::Layout::SoA);
        Tensor B(2, Tensor::Layout::SoA);
        Tensor C(4, Tensor::Layout::SoA);
        ucao::kernel::gp_batched_checked<3,3,0>(C, A, B);
    } catch (const std::invalid_argument&) {
        threw_batch = true;
    }

    return threw_index && threw_layout && threw_batch;
}

bool test_tape_backward() {
    using namespace ucao::kernel;
    alignas(64) float A[8]{};
    alignas(64) float B[8]{};
    alignas(64) float gA[8]{};
    alignas(64) float gB[8]{};
    alignas(64) float gC[8]{};
    A[1] = 1.0f;
    B[2] = 1.0f;
    gC[3] = 1.0f;
    tls_tape.reset();
    auto* n1 = tls_tape.record<GPTapeNode<8>>(A, gA, B, gB, gC);
    auto* n2 = tls_tape.record<GPTapeNode<8>>(A, gA, B, gB, gC);
    (void)n1;
    (void)n2;
    tls_tape.backward();
    const bool ok = gA[1] != 0.0f || gB[2] != 0.0f;
    tls_tape.reset();
    return ok && tls_tape.tail == nullptr;
}

bool test_arena_no_heap() {
    using namespace ucao::kernel;
    tls_arena.reset();
    const std::size_t before = g_allocs;
    for (int i = 0; i < 1000; ++i) {
        (void)tls_arena.alloc(64);
    }
    const std::size_t after = g_allocs;
    return (after - before) <= 1;
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
    run("test_sign_table_cl30", test_sign_table_cl30);
    run("test_fp_mul", test_fp_mul);
    run("test_fp_inv_sqrt", test_fp_inv_sqrt);
    run("test_fp_gp_cl30", test_fp_gp_cl30);
    run("test_reorthogonalize", test_reorthogonalize);
    run("test_avx512_shuffle_matches_scalar", test_avx512_shuffle_matches_scalar);
    run("test_dual_mv_gp", test_dual_mv_gp);
    run("test_vector_derivative", test_vector_derivative);
    run("test_mv_tensor_layout", test_mv_tensor_layout);
    run("test_mv_tensor_checked_contracts", test_mv_tensor_checked_contracts);
    run("test_tape_backward", test_tape_backward);
    run("test_arena_no_heap", test_arena_no_heap);
    std::cout << "PASSED: " << passed << " FAILED: " << failed << '\n';
    return failed == 0 ? 0 : 1;
}
