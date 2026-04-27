#include <unified_ml_stable.hpp>

#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <vector>

namespace {

unified_ml::DatasetView make_binary_fixture() {
    std::vector<std::vector<double>> X{{0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}};
    std::vector<double> y{0.0, 0.0, 1.0, 1.0};
    return unified_ml::DatasetView(X, y, unified_ml::LearningTask::Classification);
}

} // namespace

int main() {
    unified_ml::DistillationConfig cfg;
    cfg.gate_budget = 3;
    cfg.synthesis.iterations = 64;

    const auto ds = make_binary_fixture();

    unified_ml::TabularModel rf(unified_ml::RandomForestSpec{});
    (void)rf.fit(ds);

    const auto summary = unified_ml::distill_to_sle(rf, ds, cfg);
    if (summary.gate_count > cfg.gate_budget) {
        std::fprintf(stderr, "gate budget violation: %zu > %zu\n", summary.gate_count, cfg.gate_budget);
        return 1;
    }
    if (summary.artifact.feature_quantizers.size() != ds.cols()) {
        std::fprintf(stderr, "quantizer mapping size mismatch\n");
        return 2;
    }

    std::vector<double> row{1.0, 0.0};
    const auto pred = unified_ml::predict_with_distilled_sle(
        std::span<const double>(row.data(), row.size()), summary.artifact);
    if (pred != 0.0 && pred != 1.0) {
        std::fprintf(stderr, "unexpected prediction value\n");
        return 3;
    }

    const std::filesystem::path artifact_path = std::filesystem::temp_directory_path() / "sle_distilled_artifact.bin";
    unified_ml::save_distilled_artifact(summary.artifact, artifact_path.string());
    const auto loaded = unified_ml::load_distilled_artifact(artifact_path.string());

    const auto features = unified_ml::DatasetView::to_nested_vectors(ds.features());
    const auto original_batch = unified_ml::predict_with_distilled_sle(
        std::span<const std::vector<double>>(features.data(), features.size()), summary.artifact);
    const auto loaded_batch = unified_ml::predict_with_distilled_sle(
        std::span<const std::vector<double>>(features.data(), features.size()), loaded);
    if (original_batch != loaded_batch) {
        std::fprintf(stderr, "round-trip parity mismatch\n");
        return 4;
    }

    bool saw_invalid = false;
    try {
        std::vector<std::vector<double>> X{{1.0}, {1.0}};
        std::vector<double> y{1.0, 1.0};
        unified_ml::DatasetView svm_ds(X, y, unified_ml::LearningTask::Classification);
        svm::SVM svm_model;
        (void)unified_ml::distill_to_sle(unified_ml::ModelArtifact(std::move(svm_model)), svm_ds, cfg);
    } catch (const std::invalid_argument&) {
        saw_invalid = true;
    }

    if (!saw_invalid) {
        std::fprintf(stderr, "expected SVM distillation rejection\n");
        return 5;
    }

    std::filesystem::remove(artifact_path);
    return 0;
}
