# unified_ml Conan recipe         
# Usage:
# conan create . --version 1.0.0 -s build_type=Release
# # Then in your project:
# conan install . --output-folder=build --build=missing
# cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=build/conan_toolchain.cmake
#
# Publish to a Conan remote:
# conan upload unified_ml/1.0.0 -r <remote>
from conan import ConanFile
from conan.tools.cmake import CMake, CMakeToolchain, CMakeDeps, cmake_layout
from conan.tools.files import copy
import os

class UnifiedMLConan(ConanFile):
  name    = "unified_ml"
  version   = "1.0.0"
  description = "HPC-Grade C++ Machine Learning Engine"
  topics  = ("machine-learning", "hpc", "avx512", "c++17", "no-deps")
  url   = "https://github.com/conan-io/conan-center-index"  # update when submitted
  homepage  = "https://github.com/your-org/unified_ml"
  license   = "MIT"
  package_type = "library"

  # Settings & options         
  settings = "os", "compiler", "build_type", "arch"
  options  = {
    "shared":   [True, False],  # default shared=False for safety
    "fPIC":   [True, False],
    "with_openmp":  [True, False],
    "with_avx":   [True, False],
  }
  default_options = {
    "shared":   False,
    "fPIC":   True,
    "with_openmp":  True,
    "with_avx":   True,
  }

  # Source           
  exports_sources = (
    "CMakeLists.txt", "src/*", "include/*",
    "cmake/*", "examples/*", "LICENSE"
  )

  def config_options(self):
    if self.settings.os == "Windows":
    del self.options.fPIC

  def configure(self):
    if self.options.shared:
    self.options.rm_safe("fPIC")

  def layout(self):
    cmake_layout(self)

  def generate(self):
    tc = CMakeToolchain(self)
    tc.variables["UNIFIED_ML_BUILD_SHARED"]   = self.options.shared
    tc.variables["UNIFIED_ML_BUILD_STATIC"]   = not self.options.shared
    tc.variables["UNIFIED_ML_BUILD_TESTS"]  = False
    tc.variables["UNIFIED_ML_BUILD_BENCHMARKS"] = False
    tc.variables["UNIFIED_ML_ENABLE_OPENMP"]  = self.options.with_openmp
    tc.variables["UNIFIED_ML_ENABLE_AVX"]   = self.options.with_avx
    tc.variables["UNIFIED_ML_NATIVE_MARCH"]   = False # never native in packaged builds
    tc.variables["UNIFIED_ML_INSTALL"]    = True
    tc.variables["UNIFIED_ML_VERSIONED_INSTALL"] = False  # conan handles layout
    tc.generate()

    deps = CMakeDeps(self)
    deps.generate()

  def build(self):
    cmake = CMake(self)
    cmake.configure()
    cmake.build()

  def package(self):
    cmake = CMake(self)
    cmake.install()
    # Conan convention: license file
    copy(self, "LICENSE", self.source_folder, os.path.join(self.package_folder, "licenses"))

  def package_info(self):
    # CMake target info
    self.cpp_info.set_property("cmake_file_name", "unified_ml")
    self.cpp_info.set_property("cmake_target_name", "unified_ml::unified_ml")
    self.cpp_info.set_property("pkg_config_name", "unified_ml")

    # Libraries
    if self.options.shared and self.settings.os == "Windows":
    self.cpp_info.libs = ["unified_ml"]
    elif not self.options.shared and self.settings.os == "Windows":
    self.cpp_info.libs = ["unified_ml_static"]
    else:
    self.cpp_info.libs = ["unified_ml"]

    # C++17 requirement propagated to consumers
    self.cpp_info.cxxflags = ["-std=c++17"]
    if self.settings.compiler == "msvc":
    self.cpp_info.cxxflags = ["/std:c++17"]

    # System dependencies
    if self.settings.os in ["Linux", "FreeBSD"]:
    self.cpp_info.system_libs = ["m", "pthread"]
    if self.options.with_openmp:
      self.cpp_info.system_libs.append("gomp")
    elif self.settings.os == "Macos" and self.options.with_openmp:
    # libomp from Homebrew
    self.cpp_info.system_libs = ["omp"]

  def package_id(self):
    # ABI contract: minor/patch changes are ABI-compatible
    # So 1.0.0 and 1.2.3 produce the same package_id (won't re-download)
    self.info.options.rm_safe("fPIC")
