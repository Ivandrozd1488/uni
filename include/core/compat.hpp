// compat.hpp — Cross-platform compatibility macros for MSVC / GCC / Clang.
//
// Fixes three MSVC issues:
//   1. __restrict__ → __restrict (MSVC keyword)
//   2. std::aligned_alloc → _aligned_malloc/_aligned_free
//   3. #pragma omp simd → requires /openmp:experimental on MSVC

#pragma once

#include <cstddef>
#include <cstdlib>
#include <new>

#define HPC_STRINGIFY_INNER(x) #x
#define HPC_STRINGIFY(x) HPC_STRINGIFY_INNER(x)

//   restrict qualifier                             
// MSVC uses __restrict, GCC/Clang use __restrict__
#ifdef _MSC_VER
  #define HPC_RESTRICT __restrict
#else
  #define HPC_RESTRICT __restrict__
#endif

//   Aligned allocation
// MSVC lacks std::aligned_alloc (C11). Use _aligned_malloc/_aligned_free.
// GCC/Clang: use ::operator new(size, align_val_t) instead of std::aligned_alloc so that
// MemorySanitizer (MSan) correctly tracks initialization of the returned block.
inline void* hpc_aligned_alloc(std::size_t alignment, std::size_t size) {
    if (size == 0) return nullptr;
#ifdef _MSC_VER
    void* p = _aligned_malloc(size, alignment);
    if (!p) throw std::bad_alloc();
    return p;
#else
    // Round size up to a multiple of alignment (required by aligned allocators).
    const std::size_t aligned_size = ((size + alignment - 1) / alignment) * alignment;
    // operator new(size, align_val_t) is tracked by MSan; std::aligned_alloc is not.
    return ::operator new(aligned_size, std::align_val_t{alignment});
#endif
}

// alignment must match the value passed to hpc_aligned_alloc for the same pointer.
inline void hpc_aligned_free(void* p, std::size_t alignment = 64) {
    if (!p) return;
#ifdef _MSC_VER
    (void)alignment;
    _aligned_free(p);
#else
    // Must use operator delete with the same align_val_t used at allocation time.
    ::operator delete(p, std::align_val_t{alignment});
#endif
}

//   SIMD pragma                                
// MSVC requires /openmp:experimental for #pragma omp simd.
// If not available, we just skip the pragma (code still works, just no hint).
// Usage: HPC_PRAGMA_OMP_SIMD before a for loop.
#if defined(_MSC_VER)
  // MSVC OpenMP front-end support for `simd reduction` is limited unless using /openmp:llvm.
  // Keep this a no-op to avoid warning C4849 on default MSVC OpenMP.
  #define HPC_PRAGMA_OMP_SIMD
  #define HPC_PRAGMA_OMP_SIMD_REDUCTION(op, var)
#elif defined(_OPENMP)
  #define HPC_PRAGMA_OMP_SIMD _Pragma("omp simd")
  #define HPC_PRAGMA_OMP_SIMD_REDUCTION(op, var) _Pragma(HPC_STRINGIFY(omp simd reduction(op : var)))
#else
  #define HPC_PRAGMA_OMP_SIMD
  #define HPC_PRAGMA_OMP_SIMD_REDUCTION(op, var)
#endif
