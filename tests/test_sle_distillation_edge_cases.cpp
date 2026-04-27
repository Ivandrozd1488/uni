#include <models/sle/distillation.hpp>
#include <unified_ml_distillation.hpp>

#include <cstdlib>
#include <iostream>
#include <vector>

int main() {
    core::models::sle::DistillConfig cfg;
    core::models::sle::DistillDatasetBuilder builder(cfg);

    {
        std::vector<std::vector<double>> features(100, std::vector<double>{0.0, 1.0, 0.0, 1.0});
        std::vector<double> predictions(100, 1.0);
        auto ds = builder.build(std::span<const std::vector<double>>(features.data(), features.size()),
                                std::span<const double>(predictions.data(), predictions.size()));
        if (ds.examples.empty() || ds.examples.size() > 64) {
            std::cerr << "unexpected dataset size for all-positive predictions: " << ds.examples.size() << "\n";
            return 1;
        }
    }

    {
        std::vector<std::vector<double>> features(100, std::vector<double>{0.0, 1.0, 0.0, 1.0});
        std::vector<double> predictions(100, 0.0);
        auto ds = builder.build(std::span<const std::vector<double>>(features.data(), features.size()),
                                std::span<const double>(predictions.data(), predictions.size()));
        if (ds.examples.empty() || ds.examples.size() > 64) {
            std::cerr << "unexpected dataset size for all-negative predictions: " << ds.examples.size() << "\n";
            return 2;
        }
    }

    {
        unified_ml::DistilledArtifact artifact;
        artifact.metadata.artifact_version = 999;
        bool saw_version_error = false;
        try {
            std::vector<double> sample{0.0};
            (void)unified_ml::predict_with_distilled_sle(std::span<const double>(sample.data(), sample.size()), artifact);
        } catch (const std::invalid_argument&) {
            saw_version_error = true;
        }
        if (!saw_version_error) {
            std::cerr << "expected version mismatch error\n";
            return 3;
        }
    }

    {
        unified_ml::DistillationConfig run_cfg;
        run_cfg.gate_budget = 3;
        run_cfg.synthesis.iterations = 64;

        std::vector<std::vector<double>> features{{0.0}, {1.0}, {0.0}, {1.0}};
        std::vector<double> labels{0.0, 1.0, 0.0, 1.0};
        unified_ml::DatasetView ds(features, labels, unified_ml::LearningTask::Classification);
        rf::RandomForestParams params;
        params.n_estimators = 3;
        params.max_depth = 3;
        rf::RandomForest model(params);
        model.fit(ds.to_rf_dataset(rf::TaskType::Classification));

        for (std::size_t i = 0; i < 10'000; ++i) {
            const auto summary = unified_ml::distill_to_sle(model, ds, run_cfg);
            if (summary.gate_count > run_cfg.gate_budget) {
                std::cerr << "gate budget violation in stress loop at iter=" << i << "\n";
                return 4;
            }
        }
    }

    std::cout << "ok\n";
    return 0;
}
