/**
 * @file main.cpp
 * @brief Minimal installed-package smoke test for the unified_ml SDK.
 */
#include <unified_ml>

#include <cassert>
#include <iostream>

int main() {
  static_assert(UNIFIED_ML_VERSION >= UNIFIED_ML_MAKE_VERSION(1, 0, 0),
      "unified_ml >= 1.0.0 required");
  std::cout << "unified_ml " UNIFIED_ML_VERSION_STRING " install test\n";

  // PCA: simple, no external deps, exercises the full build
  core::Matrix X = core::Matrix::from_rows({
    core::Vector({1.0, 2.0}),
    core::Vector({3.0, 4.0}),
    core::Vector({5.0, 6.0})
  });
  pca::PCA p(1);
  p.fit(X);
  auto Xt = p.transform(X);
  assert(Xt.rows() == 3 && Xt.cols() == 1);
  std::cout << "PASSED\n";
  return 0;
}
