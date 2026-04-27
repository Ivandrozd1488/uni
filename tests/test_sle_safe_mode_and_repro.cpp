#include <unified_ml_stable.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

[[nodiscard]] unified_ml::DatasetView make_fixture() {
    std::vector<std::vector<double>> X{{0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}, {1.0, 1.0}, {0.0, 0.0}};
    std::vector<double> y{0.0, 0.0, 1.0, 1.0, 1.0, 0.0};
    return unified_ml::DatasetView(X, y, unified_ml::LearningTask::Classification);
}

[[nodiscard]] std::uint64_t fnv1a_mix(std::uint64_t hash, std::uint64_t value) {
    constexpr std::uint64_t k_offset = 1469598103934665603ULL;
    constexpr std::uint64_t k_prime = 1099511628211ULL;
    if (hash == 0) hash = k_offset;
    hash ^= value;
    hash *= k_prime;
    return hash;
}

[[nodiscard]] std::uint64_t artifact_fingerprint(const unified_ml::DistillationSummary& summary,
                                                 const std::vector<std::vector<double>>& probe_rows) {
    std::uint64_t h = 0;
    h = fnv1a_mix(h, static_cast<std::uint64_t>(summary.gate_count));
    h = fnv1a_mix(h, static_cast<std::uint64_t>(summary.artifact.feature_quantizers.size()));

    for (const auto& q : summary.artifact.feature_quantizers) {
        const auto scaled = static_cast<std::int64_t>(std::llround(q.threshold * 1'000'000.0));
        h = fnv1a_mix(h, static_cast<std::uint64_t>(q.source_feature_index));
        h = fnv1a_mix(h, static_cast<std::uint64_t>(scaled));
    }

    const auto preds = unified_ml::predict_with_distilled_sle(
        std::span<const std::vector<double>>(probe_rows.data(), probe_rows.size()), summary.artifact);
    for (const double p : preds) {
        h = fnv1a_mix(h, static_cast<std::uint64_t>(std::llround(p)));
    }
    return h;
}

} // namespace

int main() {
    const auto ds = make_fixture();

    unified_ml::DistillationConfig safe_cfg = unified_ml::make_sle_safe_mode_profile();
    if (safe_cfg.gate_budget != 16 || safe_cfg.synthesis.iterations != 128 || safe_cfg.synthesis.prefer_jit ||
        safe_cfg.synthesis.rollout_budget != 32 || safe_cfg.synthesis.rollout_patience != 8) {
        std::fprintf(stderr, "safe mode profile does not match conservative contract\n");
        return 1;
    }

    rf::RandomForestParams params;
    params.n_estimators = 3;
    params.max_depth = 3;
    params.random_seed = 77;
    params.bootstrap = false;
    rf::RandomForest model(params);
    model.fit(ds.to_rf_dataset(rf::TaskType::Classification));

    const auto features = unified_ml::DatasetView::to_nested_vectors(ds.features());

    const auto run_a = unified_ml::distill_to_sle(model, ds, safe_cfg);
    const auto run_b = unified_ml::distill_to_sle(model, ds, safe_cfg);

    const auto fp_a = artifact_fingerprint(run_a, features);
    const auto fp_b = artifact_fingerprint(run_b, features);
    if (fp_a != fp_b) {
        std::fprintf(stderr, "non-deterministic distillation fingerprint: %llu vs %llu\n",
                     static_cast<unsigned long long>(fp_a),
                     static_cast<unsigned long long>(fp_b));
        return 2;
    }

    return 0;
}
