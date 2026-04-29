#include <unified_ml_stable.hpp>

#include <filesystem>
#include <string>
#include <vector>

namespace {
std::filesystem::path temp_file(const std::string& name) {
    return std::filesystem::temp_directory_path() / name;
}
}

int main() {
    std::vector<std::vector<double>> X{{0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}};
    std::vector<double> y{0.0, 1.0, 1.0, 0.0};

    unified_ml::DatasetView dataset(X, y, unified_ml::LearningTask::Classification);

    rf::RandomForestParams params;
    params.n_estimators = 4;
    params.max_depth = 3;
    params.compute_oob = false;

    rf::RandomForest forest(params);
    forest.fit(dataset.to_rf_dataset(rf::TaskType::Classification));

    auto raw_predictions = forest.predict(dataset.to_rf_dataset(rf::TaskType::Classification));
    auto summary = unified_ml::evaluate(dataset, raw_predictions, 2);
    if (summary.task != unified_ml::MetricTask::Classification) return 1;

    unified_ml::ModelArtifact artifact(std::move(forest));
    auto caps = artifact.capabilities();
    if (!caps.supports_serialization) return 2;
    if (!caps.supports_batch_inference) return 3;

    auto out = artifact.run(dataset);
    if (out.values.size() != X.size()) return 4;
    if (out.probabilities.size() != X.size()) return 5;

    const std::filesystem::path path = temp_file("unified_rf_artifact.bin");
    artifact.save(path.string());
    auto loaded = unified_ml::ModelArtifact::load(unified_ml::ModelKind::RandomForest, path.string());
    auto loaded_out = loaded.run(dataset);
    if (loaded_out.values.size() != X.size()) return 6;

    unified_ml::TabularModel tabular(unified_ml::RandomForestSpec{params});
    auto fit_summary = tabular.fit(dataset);
    if (fit_summary.rows != X.size()) return 7;
    if (fit_summary.task != unified_ml::LearningTask::Classification) return 8;

    auto pred_summary = tabular.predict(dataset, 2);
    if (!pred_summary.has_evaluation) return 9;
    if (pred_summary.output.values.size() != X.size()) return 10;
    if (pred_summary.evaluation.task != unified_ml::MetricTask::Classification) return 11;

    const std::filesystem::path tabular_path = temp_file("unified_rf_tabular.bin");
    tabular.save(tabular_path.string());
    auto loaded_tabular = unified_ml::TabularModel::load(unified_ml::TabularModelKind::RandomForest, tabular_path.string());
    auto loaded_tabular_pred = loaded_tabular.predict(dataset, 2);
    if (loaded_tabular_pred.output.values.size() != X.size()) return 12;

    auto explain_summary = unified_ml::explain(loaded_tabular, &dataset);
    if (!explain_summary.has_feature_importance) return 13;

    unified_ml::AdvancedModel pca_model(unified_ml::PCASpec{2});
    auto pca_fit = pca_model.fit(dataset);
    if (pca_fit.cols != 2) return 14;
    auto pca_run = pca_model.run(dataset);
    if (pca_run.transformed.rows() != X.size()) return 15;
    const std::filesystem::path pca_path = temp_file("unified_pca.bin");
    pca_model.save(pca_path.string());
    auto loaded_pca = unified_ml::AdvancedModel::load(unified_ml::AdvancedModelKind::PCA, pca_path.string());
    auto loaded_pca_run = loaded_pca.run(dataset);
    if (loaded_pca_run.transformed.rows() != X.size()) return 16;

    unified_ml::AdvancedModel iforest_model(unified_ml::IsolationForestSpec{});
    auto iforest_fit = iforest_model.fit(dataset);
    if (!iforest_fit.capabilities.supports_batch_inference) return 17;
    auto iforest_run = iforest_model.run(dataset);
    if (iforest_run.values.size() != X.size()) return 18;
    const std::filesystem::path iforest_path = temp_file("unified_iforest.bin");
    iforest_model.save(iforest_path.string());
    auto loaded_iforest = unified_ml::AdvancedModel::load(unified_ml::AdvancedModelKind::IsolationForest, iforest_path.string());
    auto loaded_iforest_run = loaded_iforest.run(dataset);
    if (loaded_iforest_run.values.size() != X.size()) return 19;

    std::vector<double> y_reg{1.0, 2.0, 2.5, 4.0};
    unified_ml::DatasetView reg_dataset(X, y_reg, unified_ml::LearningTask::Regression);
    unified_ml::AdvancedModel gp_model(unified_ml::GPSpec{});
    auto gp_fit = gp_model.fit(reg_dataset);
    if (!gp_fit.capabilities.supports_serialization) return 20;
    const std::filesystem::path gp_path = temp_file("unified_gp.bin");
    gp_model.save(gp_path.string());
    auto loaded_gp = unified_ml::AdvancedModel::load(unified_ml::AdvancedModelKind::GaussianProcess, gp_path.string());
    auto loaded_gp_run = loaded_gp.run(reg_dataset);
    if (loaded_gp_run.values.size() != X.size()) return 21;

    std::vector<std::vector<double>> xdot{{1.0, -1.0}, {0.5, -0.5}, {0.25, -0.25}, {0.1, -0.1}};
    std::vector<std::vector<double>> sindy_X{{0.0, 1.0}, {1.0, 0.0}, {2.0, -1.0}, {3.0, -2.0}};
    sindy::SINDy low_sindy;
    low_sindy.fit(sindy_X, xdot);
    const std::filesystem::path sindy_path = temp_file("unified_sindy.bin");
    low_sindy.save(sindy_path.string());
    auto loaded_sindy = sindy::SINDy::load(sindy_path.string());
    if (loaded_sindy.equations().empty()) return 22;

    unified_ml::UnifiedMLP mlp(unified_ml::MLPSpec{{8}, unified_ml::MLPActivationKind::ReLU, 1e-2, 20, true});
    auto mlp_fit = mlp.fit(dataset);
    if (mlp_fit.rows != X.size()) return 23;
    auto mlp_pred = mlp.predict(dataset, 2);
    if (mlp_pred.output.values.size() != X.size()) return 24;
    const std::filesystem::path mlp_path = temp_file("unified_mlp.bin");
    mlp.save(mlp_path.string());
    auto loaded_mlp = unified_ml::UnifiedMLP::load(mlp_path.string());
    auto loaded_mlp_pred = loaded_mlp.predict(dataset, 2);
    if (loaded_mlp_pred.output.values.size() != X.size()) return 25;

    unified_ml::DistillationConfig distill_config;
    distill_config.gate_budget = 8;
    try {
        auto distilled_rf = unified_ml::distill_to_sle(loaded_tabular, dataset, distill_config);
        if (distilled_rf.gate_count == 0) return 26;
    } catch (const std::runtime_error&) {
        // Current SLE bridge may reject some synthesized circuits. Unified facade must fail safely.
    }

    return 0;
}
