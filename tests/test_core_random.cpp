// test_core_random.cpp
// Unit tests for src/core/random.cpp
// Previously had 0% coverage. Exercises:
//   - RNG: seed, uniform_int, uniform_real, shuffle
//   - sample_without_replacement
//   - bootstrap_indices
//   - sample_population
//   - random_uniform

#include <core/random.hpp>

#include <algorithm>
#include <cassert>
#include <iostream>
#include <numeric>
#include <set>
#include <string>
#include <vector>

namespace {

int g_pass = 0, g_fail = 0;

void check(bool ok, const std::string& name) {
    if (ok) { std::cout << "PASS  " << name << "\n"; ++g_pass; }
    else    { std::cout << "FAIL  " << name << "\n"; ++g_fail; }
}

void test_rng_basics() {
    core::RNG rng(42);

    // uniform_int stays in range
    bool all_in_range = true;
    for (int i = 0; i < 200; ++i) {
        int v = rng.uniform_int(3, 7);
        if (v < 3 || v > 7) { all_in_range = false; break; }
    }
    check(all_in_range, "RNG::uniform_int range");

    // uniform_real stays in [0, 1)
    bool all_real = true;
    for (int i = 0; i < 200; ++i) {
        double v = rng.uniform_real();
        if (v < 0.0 || v >= 1.0) { all_real = false; break; }
    }
    check(all_real, "RNG::uniform_real default range");

    // custom range
    bool custom_range = true;
    for (int i = 0; i < 100; ++i) {
        double v = rng.uniform_real(5.0, 10.0);
        if (v < 5.0 || v >= 10.0) { custom_range = false; break; }
    }
    check(custom_range, "RNG::uniform_real custom range");

    // re-seed gives same sequence
    rng.seed(1);
    int a = rng.uniform_int(0, 1000);
    rng.seed(1);
    int b = rng.uniform_int(0, 1000);
    check(a == b, "RNG same seed same sequence");

    // engine() is accessible
    auto& eng = rng.engine();
    (void)eng;
    check(true, "RNG::engine accessible");
}

void test_rng_shuffle() {
    core::RNG rng(7);
    std::vector<int> v{1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<int> orig = v;

    rng.shuffle(v);
    // Shuffle shouldn't lose elements
    std::sort(v.begin(), v.end());
    check(v == orig, "RNG::shuffle preserves elements");
}

void test_sample_without_replacement() {
    core::RNG rng(99);

    // basic: k < n
    auto s = core::sample_without_replacement(10, 4, rng);
    check(s.size() == 4, "sample_without_replacement size");

    // all indices unique
    std::set<std::size_t> unique(s.begin(), s.end());
    check(unique.size() == 4, "sample_without_replacement unique");

    // all in [0, n)
    bool in_range = true;
    for (auto idx : s) if (idx >= 10) { in_range = false; break; }
    check(in_range, "sample_without_replacement range");

    // k == n: returns all indices
    auto all = core::sample_without_replacement(5, 5, rng);
    std::sort(all.begin(), all.end());
    check(all.size() == 5 && all[0] == 0 && all[4] == 4,
          "sample_without_replacement k==n");

    // k == 0: empty
    auto empty = core::sample_without_replacement(10, 0, rng);
    check(empty.empty(), "sample_without_replacement k==0 empty");
}

void test_bootstrap_indices() {
    core::RNG rng(13);

    auto idx = core::bootstrap_indices(8, rng);
    check(idx.size() == 8, "bootstrap_indices size matches n");

    // Duplicates are expected (with replacement), but all indices in [0, n)
    bool in_range = true;
    for (auto i : idx) if (i >= 8) { in_range = false; break; }
    check(in_range, "bootstrap_indices all in range");
}

void test_sample_population() {
    std::mt19937 mt(42);
    std::vector<std::size_t> pop = {10, 20, 30, 40, 50};

    auto s = core::sample_population(pop, 3, mt);
    check(s.size() == 3, "sample_population size");

    // All drawn from pop
    bool from_pop = true;
    for (auto v : s) {
        bool found = std::find(pop.begin(), pop.end(), v) != pop.end();
        if (!found) { from_pop = false; break; }
    }
    check(from_pop, "sample_population values from population");

    // Unique
    std::set<std::size_t> u(s.begin(), s.end());
    check(u.size() == 3, "sample_population unique");

    // k == pop.size()
    auto all = core::sample_population(pop, pop.size(), mt);
    check(all.size() == pop.size(), "sample_population k==size");
}

void test_random_uniform() {
    std::mt19937 mt(7);

    bool ok = true;
    for (int i = 0; i < 500; ++i) {
        double v = core::random_uniform(-3.0, 3.0, mt);
        if (v < -3.0 || v >= 3.0) { ok = false; break; }
    }
    check(ok, "random_uniform range [-3, 3)");
}

void test_iota_indices() {
    auto idx = core::iota_indices(5);
    check(idx.size() == 5, "iota_indices size");
    check(idx[0] == 0 && idx[4] == 4, "iota_indices values");

    auto empty = core::iota_indices(0);
    check(empty.empty(), "iota_indices size 0");
}

} // namespace

int main() {
    test_rng_basics();
    test_rng_shuffle();
    test_sample_without_replacement();
    test_bootstrap_indices();
    test_sample_population();
    test_random_uniform();
    test_iota_indices();

    std::cout << "\n" << g_pass << " passed, " << g_fail << " failed\n";
    return g_fail ? 1 : 0;
}
