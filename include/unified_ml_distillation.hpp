#pragma once

#include "models/iforest/isolation_forest.hpp"
#include "models/mlp/model.hpp"
#include "models/pca/pca.hpp"
#include "models/pinn/neural_network.hpp"
#include "models/rf/random_forest.hpp"
#include "models/sle/distillation.hpp"
#include "models/xgboost/xgboost_enhanced.hpp"
#include "sle_backend.hpp"
#include "unified_ml_artifact.hpp"
#include "unified_ml_capabilities.hpp"
#include "unified_ml_dataset.hpp"
#include "unified_ml_tabular.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include "core/span_compat.hpp"
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace unified_ml {

using DistillationConfig = core::models::sle::DistillConfig;
using DistillationDataset = core::models::sle::DistillDataset;
using DistillationDatasetBuilder = core::models::sle::DistillDatasetBuilder;
using DistilledCircuit = core::models::sle::BooleanCascade;

enum class DistilledBinarizationScheme : std::uint32_t {
    ThresholdGreaterEqual = 1,
};

struct DistilledFeatureQuantizer {
    std::size_t source_feature_index = 0;
    double threshold = 0.0;
    DistilledBinarizationScheme scheme = DistilledBinarizationScheme::ThresholdGreaterEqual;
};

struct DistilledTargetPolicy {
    double threshold = 0.5;
    bool positive_when_greater_equal = true;
};

struct DistilledArtifactMetadata {
    std::uint32_t artifact_version = 1;
    std::string distiller = "unified_ml.sle.distillation";
};

struct DistilledArtifact {
    DistilledCircuit circuit{};
    std::vector<DistilledFeatureQuantizer> feature_quantizers;
    DistilledTargetPolicy target_policy{};
    DistilledArtifactMetadata metadata{};

    [[nodiscard]] std::size_t gate_count() const noexcept { return circuit.gate_count(); }
    [[nodiscard]] std::size_t input_count() const noexcept { return circuit.input_count(); }
};

struct DistillationSummary {
    DistilledCircuit circuit{};
    std::size_t gate_count = 0;
    bool exact = false;
    DistilledArtifact artifact{};
};

[[nodiscard]] inline DistillationConfig make_sle_safe_mode_profile() {
    DistillationConfig cfg;
    cfg.gate_budget = 16;
    cfg.target_accuracy = 0.97;
    cfg.threshold = 0.5;
    cfg.synthesis.iterations = 128;
    cfg.synthesis.rollout_budget = 32;
    cfg.synthesis.rollout_patience = 8;
    cfg.synthesis.use_mdl = true;
    cfg.synthesis.prefer_jit = false;
    cfg.synthesis.seed = 0x5AFE0001ULL;
    return cfg;
}

namespace detail {

inline constexpr std::uint32_t k_distilled_artifact_version = 1;

[[nodiscard]] inline std::vector<std::vector<double>> dataset_features(const DatasetView& dataset) {
    return DatasetView::to_nested_vectors(dataset.features());
}

[[nodiscard]] inline DistillationSummary build_summary(core::models::sle::DistillDataset dataset,
                                                       DistilledCircuit circuit,
                                                       double target_threshold) {
    DistillationSummary summary;
    summary.circuit = circuit;
    summary.gate_count = summary.circuit.gate_count();
    summary.exact = false;

    summary.artifact.circuit = summary.circuit;
    summary.artifact.target_policy.threshold = target_threshold;
    summary.artifact.target_policy.positive_when_greater_equal = true;
    summary.artifact.metadata.artifact_version = k_distilled_artifact_version;
    summary.artifact.feature_quantizers.reserve(dataset.thresholds.size());
    for (std::size_t i = 0; i < dataset.thresholds.size(); ++i) {
        summary.artifact.feature_quantizers.push_back(DistilledFeatureQuantizer{
            .source_feature_index = i,
            .threshold = dataset.thresholds[i],
            .scheme = DistilledBinarizationScheme::ThresholdGreaterEqual,
        });
    }

    return summary;
}

[[nodiscard]] inline double target_threshold_from_predictions(std::span<const double> predictions) {
    if (predictions.empty()) {
        throw std::invalid_argument("distill_to_sle: predictions are empty");
    }
    std::vector<double> sorted(predictions.begin(), predictions.end());
    std::sort(sorted.begin(), sorted.end());
    sorted.erase(std::unique(sorted.begin(), sorted.end()), sorted.end());
    if (sorted.size() == 1) return sorted.front();

    const auto midpoint = sorted.size() / 2;
    if (sorted.size() % 2 == 0) {
        return 0.5 * (sorted[midpoint - 1] + sorted[midpoint]);
    }
    return sorted[midpoint];
}

[[nodiscard]] inline std::vector<double> model_predictions(const rf::RandomForest& model,
                                                         std::span<const std::vector<double>> features) {
    return model.predict(std::vector<std::vector<double>>(features.begin(), features.end()));
}

[[nodiscard]] inline std::vector<double> model_predictions(const xgb::XGBModel& model,
                                                         std::span<const std::vector<double>> features) {
    std::vector<std::vector<float>> xgb_features;
    xgb_features.reserve(features.size());
    for (const auto& row : features) {
        xgb_features.emplace_back(row.begin(), row.end());
    }
    const auto preds_f = model.predict(xgb_features);
    return std::vector<double>(preds_f.begin(), preds_f.end());
}

template <class Model>
[[nodiscard]] inline std::vector<double> model_predictions(const Model& model,
                                                         std::span<const std::vector<double>> features) {
    const auto raw = model.predict(std::vector<std::vector<double>>(features.begin(), features.end()));
    std::vector<double> out;
    out.reserve(raw.size());
    for (const auto value : raw) out.push_back(static_cast<double>(value));
    return out;
}

template <class Model>
[[nodiscard]] inline DistillationSummary distill_model_impl(const Model& model,
                                                            const DatasetView& dataset,
                                                            const DistillationConfig& config) {
    const auto features = dataset_features(dataset);
    const auto predictions = model_predictions(model, std::span<const std::vector<double>>(features.data(), features.size()));


    DistillationDatasetBuilder builder(config);
    auto distill_dataset = builder.build(std::span<const std::vector<double>>(features.data(), features.size()),
                                         std::span<const double>(predictions.data(), predictions.size()));
    auto circuit = core::models::sle::distill_to_circuit(distill_dataset, config);
    return build_summary(std::move(distill_dataset), std::move(circuit),
                         target_threshold_from_predictions(std::span<const double>(predictions.data(), predictions.size())));
}

[[nodiscard]] inline std::vector<::sle::BitVector> quantize_row(
    std::span<const double> raw_features,
    const DistilledArtifact& artifact) {
    std::vector<::sle::BitVector> inputs;
    inputs.reserve(artifact.feature_quantizers.size());

    for (const auto& quantizer : artifact.feature_quantizers) {
        if (quantizer.scheme != DistilledBinarizationScheme::ThresholdGreaterEqual) {
            throw std::invalid_argument("predict_with_distilled_sle: unsupported quantization scheme");
        }
        if (quantizer.source_feature_index >= raw_features.size()) {
            throw std::invalid_argument("predict_with_distilled_sle: missing source feature for quantization mapping");
        }

        ::sle::BitVector bit(1);
        const bool v = raw_features[quantizer.source_feature_index] >= quantizer.threshold;
        bit.set(0, v);
        inputs.push_back(std::move(bit));
    }

    return inputs;
}

template <class T>
inline void write_raw(std::ostream& out, const T& value) {
    out.write(reinterpret_cast<const char*>(&value), static_cast<std::streamsize>(sizeof(T)));
    if (!out) throw std::runtime_error("save_distilled_artifact: write failure");
}

template <class T>
[[nodiscard]] inline T read_raw(std::istream& in) {
    T value{};
    in.read(reinterpret_cast<char*>(&value), static_cast<std::streamsize>(sizeof(T)));
    if (!in) throw std::runtime_error("load_distilled_artifact: truncated payload");
    return value;
}

inline void write_string(std::ostream& out, const std::string& value) {
    const auto n = static_cast<std::uint64_t>(value.size());
    write_raw(out, n);
    out.write(value.data(), static_cast<std::streamsize>(value.size()));
    if (!out) throw std::runtime_error("save_distilled_artifact: write failure");
}

[[nodiscard]] inline std::string read_string(std::istream& in) {
    const auto n = read_raw<std::uint64_t>(in);
    std::string value(n, '\0');
    in.read(value.data(), static_cast<std::streamsize>(n));
    if (!in) throw std::runtime_error("load_distilled_artifact: truncated payload");
    return value;
}

} // namespace detail

[[nodiscard]] inline DistillationSummary distill_to_sle(const ModelArtifact& artifact,
                                                        const DatasetView& dataset,
                                                        const DistillationConfig& config = {}) {
    return std::visit([&](const auto& model) -> DistillationSummary {
        using Model = std::decay_t<decltype(model)>;
        if constexpr (std::is_same_v<Model, svm::SVM>) {
            throw std::invalid_argument("distill_to_sle: SVM artifacts are not supported for SLE distillation");
        } else {
            return detail::distill_model_impl(model, dataset, config);
        }
    }, artifact.variant());
}

template <typename Model>
[[nodiscard]] inline DistillationSummary distill_to_sle(const Model& model,
                                                        const DatasetView& dataset,
                                                        const DistillationConfig& config = {}) {
    return detail::distill_model_impl(model, dataset, config);
}

[[nodiscard]] inline DistillationSummary distill_to_sle(const TabularModel& model,
                                                        const DatasetView& dataset,
                                                        const DistillationConfig& config = {}) {
    const auto* artifact = model.artifact();
    if (!artifact) {
        throw std::runtime_error("distill_to_sle(TabularModel): model is not fitted");
    }
    return distill_to_sle(*artifact, dataset, config);
}

[[nodiscard]] inline DistillationSummary distill_to_sle(const mlp::Model& model) {
    DistillationSummary summary;
    summary.circuit = sle_backend::distill_to_logic(model);
    summary.gate_count = summary.circuit.gate_count();
    summary.exact = true;
    summary.artifact.circuit = summary.circuit;
    summary.artifact.metadata.artifact_version = detail::k_distilled_artifact_version;
    return summary;
}

[[nodiscard]] inline DistillationSummary distill_to_sle(const pinn::NeuralNetwork& model) {
    DistillationSummary summary;
    summary.circuit = sle_backend::distill_to_logic(model);
    summary.gate_count = summary.circuit.gate_count();
    summary.exact = true;
    summary.artifact.circuit = summary.circuit;
    summary.artifact.metadata.artifact_version = detail::k_distilled_artifact_version;
    return summary;
}

[[nodiscard]] inline double predict_with_distilled_sle(std::span<const double> raw_features,
                                                       const DistilledArtifact& artifact) {
    if (artifact.metadata.artifact_version != detail::k_distilled_artifact_version) {
        throw std::invalid_argument("predict_with_distilled_sle: artifact version mismatch");
    }
    if (artifact.circuit.input_count() != artifact.feature_quantizers.size()) {
        throw std::invalid_argument("predict_with_distilled_sle: quantization mapping size mismatch");
    }

    const auto inputs = detail::quantize_row(raw_features, artifact);
    const auto output = artifact.circuit.evaluate(inputs);
    const bool positive = output.get(0);
    const bool ge_positive = artifact.target_policy.positive_when_greater_equal;
    return (positive == ge_positive) ? 1.0 : 0.0;
}

[[nodiscard]] inline std::vector<double> predict_with_distilled_sle(
    std::span<const std::vector<double>> raw_features,
    const DistilledArtifact& artifact) {
    std::vector<double> predictions;
    predictions.reserve(raw_features.size());
    for (const auto& row : raw_features) {
        predictions.push_back(predict_with_distilled_sle(std::span<const double>(row.data(), row.size()), artifact));
    }
    return predictions;
}

inline void save_distilled_artifact(const DistilledArtifact& artifact, const std::string& path) {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("save_distilled_artifact: cannot open file");

    constexpr std::array<char, 8> magic{{'U', 'M', 'L', 'S', 'L', 'E', '1', '\0'}};
    out.write(magic.data(), static_cast<std::streamsize>(magic.size()));
    if (!out) throw std::runtime_error("save_distilled_artifact: write failure");

    detail::write_raw(out, artifact.metadata.artifact_version);
    detail::write_string(out, artifact.metadata.distiller);

    detail::write_raw(out, static_cast<std::uint64_t>(artifact.feature_quantizers.size()));
    for (const auto& quantizer : artifact.feature_quantizers) {
        detail::write_raw(out, static_cast<std::uint64_t>(quantizer.source_feature_index));
        detail::write_raw(out, quantizer.threshold);
        detail::write_raw(out, static_cast<std::uint32_t>(quantizer.scheme));
    }

    detail::write_raw(out, artifact.target_policy.threshold);
    detail::write_raw(out, static_cast<std::uint8_t>(artifact.target_policy.positive_when_greater_equal ? 1 : 0));

    detail::write_raw(out, static_cast<std::uint64_t>(artifact.circuit.input_count()));
    detail::write_raw(out, static_cast<std::uint64_t>(artifact.circuit.gate_count()));
    for (const auto& gate : artifact.circuit.gates()) {
        detail::write_raw(out, static_cast<std::uint64_t>(gate.a));
        detail::write_raw(out, static_cast<std::uint64_t>(gate.b));
        detail::write_raw(out, static_cast<std::uint64_t>(gate.c));
        detail::write_raw(out, gate.mask);
    }
}

[[nodiscard]] inline DistilledArtifact load_distilled_artifact(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("load_distilled_artifact: cannot open file");

    constexpr std::array<char, 8> magic{{'U', 'M', 'L', 'S', 'L', 'E', '1', '\0'}};
    std::array<char, 8> file_magic{};
    in.read(file_magic.data(), static_cast<std::streamsize>(file_magic.size()));
    if (!in || file_magic != magic) {
        throw std::runtime_error("load_distilled_artifact: invalid magic");
    }

    DistilledArtifact artifact;
    artifact.metadata.artifact_version = detail::read_raw<std::uint32_t>(in);
    if (artifact.metadata.artifact_version != detail::k_distilled_artifact_version) {
        throw std::invalid_argument("load_distilled_artifact: unsupported artifact version");
    }
    artifact.metadata.distiller = detail::read_string(in);

    const auto quantizer_count = detail::read_raw<std::uint64_t>(in);
    artifact.feature_quantizers.reserve(static_cast<std::size_t>(quantizer_count));
    for (std::uint64_t i = 0; i < quantizer_count; ++i) {
        const auto source_feature_index = detail::read_raw<std::uint64_t>(in);
        const auto threshold = detail::read_raw<double>(in);
        const auto scheme_raw = detail::read_raw<std::uint32_t>(in);
        if (scheme_raw != static_cast<std::uint32_t>(DistilledBinarizationScheme::ThresholdGreaterEqual)) {
            throw std::invalid_argument("load_distilled_artifact: unsupported quantization scheme");
        }

        artifact.feature_quantizers.push_back(DistilledFeatureQuantizer{
            .source_feature_index = static_cast<std::size_t>(source_feature_index),
            .threshold = threshold,
            .scheme = DistilledBinarizationScheme::ThresholdGreaterEqual,
        });
    }

    artifact.target_policy.threshold = detail::read_raw<double>(in);
    artifact.target_policy.positive_when_greater_equal = detail::read_raw<std::uint8_t>(in) != 0;

    const auto input_count = detail::read_raw<std::uint64_t>(in);
    const auto gate_count = detail::read_raw<std::uint64_t>(in);
    artifact.circuit = DistilledCircuit(static_cast<std::size_t>(input_count));
    for (std::uint64_t i = 0; i < gate_count; ++i) {
        ::sle::TernaryGate gate;
        gate.a = static_cast<std::size_t>(detail::read_raw<std::uint64_t>(in));
        gate.b = static_cast<std::size_t>(detail::read_raw<std::uint64_t>(in));
        gate.c = static_cast<std::size_t>(detail::read_raw<std::uint64_t>(in));
        gate.mask = detail::read_raw<std::uint8_t>(in);
        artifact.circuit.add_gate(gate);
    }

    if (artifact.circuit.input_count() != artifact.feature_quantizers.size()) {
        throw std::invalid_argument("load_distilled_artifact: quantization mapping size mismatch");
    }

    return artifact;
}

} // namespace unified_ml
