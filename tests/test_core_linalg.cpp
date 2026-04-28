// test_core_linalg.cpp
// Unit tests for src/core/linalg.cpp
// Previously had 19% coverage. This file exercises the uncovered paths:
//   - Vector arithmetic (operator+, -, *, dot, norm, normalized)
//   - Matrix construction, identity, from_rows, arithmetic, transpose
//   - Matrix::solve (Gaussian elimination)
//   - jacobi_eigen / sort_eigen
//   - AlignedVec copy/move semantics

#include <core/linalg.hpp>

#include <cassert>
#include <cmath>
#include <iostream>
#include <string>

namespace {

int g_pass = 0, g_fail = 0;

void check(bool ok, const std::string& name) {
    if (ok) { std::cout << "PASS  " << name << "\n"; ++g_pass; }
    else    { std::cout << "FAIL  " << name << "\n"; ++g_fail; }
}

constexpr double EPS = 1e-9;
bool near(double a, double b, double tol = EPS) { return std::fabs(a - b) < tol; }

// ── AlignedVec ───────────────────────────────────────────────────────────────
void test_aligned_vec() {
    core::AlignedVec a(4, 1.0);
    check(a.size() == 4 && a[0] == 1.0, "AlignedVec ctor + index");

    core::AlignedVec b(a);                      // copy
    b[0] = 9.0;
    check(a[0] == 1.0 && b[0] == 9.0, "AlignedVec copy independence");

    core::AlignedVec c(std::move(b));            // move
    check(c[0] == 9.0, "AlignedVec move value");

    core::AlignedVec d;
    d = a;                                       // copy-assign
    check(d[1] == 1.0, "AlignedVec copy-assign");

    core::AlignedVec e;
    e = std::move(c);                            // move-assign
    check(e[0] == 9.0, "AlignedVec move-assign");

    core::AlignedVec empty;
    check(empty.size() == 0 && empty.data() == nullptr, "AlignedVec default empty");
}

// ── Vector ───────────────────────────────────────────────────────────────────
void test_vector_basics() {
    core::Vector v(3, 2.0);
    check(v.size() == 3 && v[0] == 2.0, "Vector ctor");
    check(!v.empty(), "Vector not empty");

    core::Vector z;
    check(z.empty(), "Vector default empty");

    core::Vector il{1.0, 2.0, 3.0};
    check(il.size() == 3 && il[2] == 3.0, "Vector initializer_list");

    // at() bounds check
    bool threw = false;
    try { il.at(99); } catch (const std::out_of_range&) { threw = true; }
    check(threw, "Vector::at throws out_of_range");
}

void test_vector_arithmetic() {
    core::Vector a{1.0, 2.0, 3.0};
    core::Vector b{4.0, 5.0, 6.0};

    auto sum  = a + b;
    check(near(sum[0], 5.0) && near(sum[2], 9.0), "Vector operator+");

    auto diff = b - a;
    check(near(diff[0], 3.0) && near(diff[2], 3.0), "Vector operator-");

    auto scaled = a * 2.0;
    check(near(scaled[1], 4.0), "Vector operator* scalar");

    double d = a.dot(b);
    check(near(d, 1*4 + 2*5 + 3*6), "Vector dot");

    double n = a.norm();
    check(near(n, std::sqrt(14.0)), "Vector norm");

    auto nrm = a.normalized();
    check(near(nrm.norm(), 1.0, 1e-12), "Vector normalized unit length");

    // free operator scalar * vec
    auto s2 = 3.0 * a;
    check(near(s2[0], 3.0), "scalar * Vector");
}

// ── Matrix ───────────────────────────────────────────────────────────────────
void test_matrix_construction() {
    core::Matrix A(3, 4, 0.0);
    check(A.rows() == 3 && A.cols() == 4, "Matrix dimensions");
    check(!A.empty(), "Matrix not empty");

    core::Matrix E = core::Matrix::identity(3);
    check(near(E(0,0), 1.0) && near(E(0,1), 0.0) && near(E(2,2), 1.0), "Matrix identity");

    core::Matrix Z = core::Matrix::zeros(2, 3);
    check(Z.rows() == 2 && Z.cols() == 3 && near(Z(1,2), 0.0), "Matrix zeros");

    core::Vector r0{1.0, 2.0};
    core::Vector r1{3.0, 4.0};
    core::Matrix FR = core::Matrix::from_rows({r0, r1});
    check(near(FR(0,1), 2.0) && near(FR(1,0), 3.0), "Matrix from_rows");

    // row / col / set_row / set_col
    auto row1 = FR.row(1);
    check(near(row1[0], 3.0), "Matrix::row");

    auto col0 = FR.col(0);
    check(near(col0[1], 3.0), "Matrix::col");

    core::Vector nr{7.0, 8.0};
    FR.set_row(0, nr);
    check(near(FR(0,0), 7.0), "Matrix::set_row");

    core::Vector nc{5.0, 6.0};
    FR.set_col(1, nc);
    check(near(FR(1,1), 6.0), "Matrix::set_col");
}

void test_matrix_arithmetic() {
    core::Matrix A = core::Matrix::identity(2);
    core::Matrix B(2, 2, 1.0);

    auto C = A + B;
    check(near(C(0,0), 2.0) && near(C(0,1), 1.0), "Matrix operator+");

    auto D = B - A;
    check(near(D(0,0), 0.0) && near(D(0,1), 1.0), "Matrix operator-");

    // Matrix * scalar
    auto S = B * 3.0;
    check(near(S(1,1), 3.0), "Matrix * scalar");

    // Matrix / scalar
    auto Sd = S / 3.0;
    check(near(Sd(0,0), 1.0), "Matrix / scalar");

    // += and -=
    core::Matrix M = core::Matrix::identity(2);
    M += B;
    check(near(M(0,0), 2.0), "Matrix +=");
    M -= B;
    check(near(M(0,0), 1.0), "Matrix -=");

    // Matrix * Matrix (2x2 identity * B = B)
    auto AB = A * B;
    check(near(AB(0,0), 1.0) && near(AB(0,1), 1.0), "Matrix * Matrix");

    // Matrix * Vector
    core::Vector v{2.0, 3.0};
    core::Matrix M2(2, 2, 1.0);
    auto mv = M2 * v;
    check(near(mv[0], 5.0) && near(mv[1], 5.0), "Matrix * Vector");

    // free operator scalar * Matrix
    auto sm = 2.0 * A;
    check(near(sm(0,0), 2.0), "scalar * Matrix");
}

void test_matrix_transpose_submatrix() {
    core::Matrix A(2, 3, 0.0);
    A(0,0)=1; A(0,1)=2; A(0,2)=3;
    A(1,0)=4; A(1,1)=5; A(1,2)=6;

    auto T = A.transpose();
    check(T.rows() == 3 && T.cols() == 2, "transpose dimensions");
    check(near(T(0,1), 4.0) && near(T(2,0), 3.0), "transpose values");

    auto sub = A.submatrix(0, 1, 2, 3);
    check(sub.rows() == 2 && sub.cols() == 2, "submatrix dimensions");
    check(near(sub(0,0), 2.0) && near(sub(1,1), 6.0), "submatrix values");
}

void test_matrix_solve() {
    // 2x2 system: [2 1; 1 3] x = [5; 10]  → x = [1; 3]
    core::Matrix A(2, 2, 0.0);
    A(0,0)=2; A(0,1)=1;
    A(1,0)=1; A(1,1)=3;
    core::Vector b{5.0, 10.0};
    auto x = A.solve(b);
    check(near(x[0], 1.0, 1e-10) && near(x[1], 3.0, 1e-10), "Matrix::solve 2x2");

    // 3x3 identity solve → x = b
    core::Matrix I3 = core::Matrix::identity(3);
    core::Vector b3{1.0, 2.0, 3.0};
    auto x3 = I3.solve(b3);
    check(near(x3[0], 1.0) && near(x3[2], 3.0), "Matrix::solve identity");
}

// ── jacobi_eigen ─────────────────────────────────────────────────────────────
void test_jacobi_eigen() {
    // Symmetric 2x2: [3 1; 1 3] → eigenvalues 4 and 2
    core::Matrix A(2, 2, 0.0);
    A(0,0)=3; A(0,1)=1;
    A(1,0)=1; A(1,1)=3;

    auto res = core::jacobi_eigen(A);
    check(res.eigenvalues.size() == 2, "jacobi_eigen eigenvalue count");
    check(near(res.eigenvalues[0], 4.0, 1e-10) && near(res.eigenvalues[1], 2.0, 1e-10),
          "jacobi_eigen eigenvalues 4 and 2");

    // Eigenvectors orthogonal
    auto e0 = res.eigenvectors.col(0);
    auto e1 = res.eigenvectors.col(1);
    check(near(e0.dot(e1), 0.0, 1e-10), "jacobi_eigen eigenvectors orthogonal");

    // sort_eigen doesn't crash and stays sorted
    core::sort_eigen(res.eigenvalues, res.eigenvectors);
    check(res.eigenvalues[0] >= res.eigenvalues[1], "sort_eigen descending");

    // 1x1 degenerate
    core::Matrix A1(1, 1, 5.0);
    auto r1 = core::jacobi_eigen(A1);
    check(near(r1.eigenvalues[0], 5.0, 1e-10), "jacobi_eigen 1x1");
}

} // namespace

int main() {
    test_aligned_vec();
    test_vector_basics();
    test_vector_arithmetic();
    test_matrix_construction();
    test_matrix_arithmetic();
    test_matrix_transpose_submatrix();
    test_matrix_solve();
    test_jacobi_eigen();

    std::cout << "\n" << g_pass << " passed, " << g_fail << " failed\n";
    return g_fail ? 1 : 0;
}
