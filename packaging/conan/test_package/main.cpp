#include <unified_ml>
#include <iostream>
int main() {
  std::cout << "Conan package test: unified_ml " UNIFIED_ML_VERSION_STRING "\n";
  pca::PCA p(1);
  core::Matrix X = {{1.0,2.0},{3.0,4.0}};
  p.fit(X);
  std::cout << "PASSED\n";
  return 0;
}
