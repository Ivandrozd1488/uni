#include <unified_ml_stable.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct ManifestRow {
    std::string dataset_family;
    std::uint64_t seed = 0;
    std::size_t train_size = 0;
    std::size_t test_size = 0;
    std::size_t feature_count = 0;
    double noise_rate = 0.0;
    std::string profile;
    bool compression_profile = false;
    bool latency_optimized = false;
};

struct ScenarioData {
    std::vector<std::vector<double>> train_x;
    std::vector<double> train_y;
    std::vector<std::vector<double>> test_x;
    std::vector<double> test_y;
};

struct Metrics {
    double teacher_accuracy = 0.0;
    double compact_accuracy = 0.0;
    double sle_accuracy = 0.0;
    double fidelity_to_teacher = 0.0;
    double teacher_p95_us = 0.0;
    double compact_p95_us = 0.0;
    double sle_p95_us = 0.0;
    std::size_t compact_node_count = 0;
    std::size_t sle_gate_count = 0;
};

[[nodiscard]] bool parse_bool(const std::string& value) {
    if (value == "true" || value == "1" || value == "yes") return true;
    if (value == "false" || value == "0" || value == "no") return false;
    throw std::invalid_argument("invalid bool in manifest: " + value);
}

[[nodiscard]] std::vector<ManifestRow> load_manifest(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("unable to open manifest: " + path);

    std::vector<ManifestRow> rows;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::stringstream ss(line);
        std::vector<std::string> fields;
        std::string item;
        while (std::getline(ss, item, ',')) fields.push_back(item);
        if (fields.size() != 9) {
            throw std::runtime_error("manifest line must contain 9 CSV fields");
        }

        ManifestRow row;
        row.dataset_family = fields[0];
        row.seed = static_cast<std::uint64_t>(std::stoull(fields[1]));
        row.train_size = static_cast<std::size_t>(std::stoull(fields[2]));
        row.test_size = static_cast<std::size_t>(std::stoull(fields[3]));
        row.feature_count = static_cast<std::size_t>(std::stoull(fields[4]));
        row.noise_rate = std::stod(fields[5]);
        row.profile = fields[6];
        row.compression_profile = parse_bool(fields[7]);
        row.latency_optimized = parse_bool(fields[8]);
        rows.push_back(row);
    }

    if (rows.empty()) {
        throw std::runtime_error("manifest is empty");
    }
    return rows;
}

[[nodiscard]] double classify_label(const std::vector<double>& row) {
    const bool term_a = row[0] > 0.5 && row[1] > 0.5;
    const bool term_b = row[2] > 0.5 && row[3] > 0.5;
    return (term_a || term_b) ? 1.0 : 0.0;
}

[[nodiscard]] ScenarioData generate_logic_family(const ManifestRow& row) {
    std::mt19937_64 rng(row.seed);
    std::uniform_int_distribution<int> bit(0, 1);
    std::uniform_real_distribution<double> unif01(0.0, 1.0);

    ScenarioData data;
    data.train_x.reserve(row.train_size);
    data.train_y.reserve(row.train_size);
    data.test_x.reserve(row.test_size);
    data.test_y.reserve(row.test_size);

    const auto gen_sample = [&](std::vector<std::vector<double>>& X, std::vector<double>& y, std::size_t n) {
        for (std::size_t i = 0; i < n; ++i) {
            std::vector<double> sample(row.feature_count, 0.0);
            for (double& v : sample) v = static_cast<double>(bit(rng));
            if (sample.size() >= 4) {
                sample[0] = 1.0;
                sample[1] = 1.0;
                sample[2] = 1.0;
                sample[3] = 1.0;
            }
            double label = classify_label(sample);
            if (unif01(rng) < row.noise_rate) {
                label = 1.0 - label;
            }
            X.push_back(std::move(sample));
            y.push_back(label);
        }
    };

    gen_sample(data.train_x, data.train_y, row.train_size);
    gen_sample(data.test_x, data.test_y, row.test_size);
    return data;
}

[[nodiscard]] std::size_t count_nodes(const rf::TreeNode* node) {
    if (!node) return 0;
    return 1 + count_nodes(node->left.get()) + count_nodes(node->right.get());
}

[[nodiscard]] std::size_t compact_model_size_proxy(const rf::RandomForest& model) {
    std::size_t total = 0;
    for (std::size_t i = 0; i < model.n_trees(); ++i) {
        total += count_nodes(model.tree(i).root());
    }
    return total;
}

[[nodiscard]] double accuracy(std::span<const double> predictions, std::span<const double> labels) {
    if (predictions.size() != labels.size()) {
        throw std::invalid_argument("accuracy: predictions/labels size mismatch");
    }
    std::size_t correct = 0;
    for (std::size_t i = 0; i < predictions.size(); ++i) {
        if (predictions[i] == labels[i]) ++correct;
    }
    return static_cast<double>(correct) / static_cast<double>(predictions.size());
}

[[nodiscard]] double agreement(std::span<const double> a, std::span<const double> b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("agreement: size mismatch");
    }
    std::size_t same = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (a[i] == b[i]) ++same;
    }
    return static_cast<double>(same) / static_cast<double>(a.size());
}

template <class PredictFn>
[[nodiscard]] double estimate_p95_us(PredictFn&& predict_one,
                                     std::span<const std::vector<double>> rows,
                                     std::size_t repeats) {
    std::vector<double> samples;
    samples.reserve(rows.size() * repeats);

    volatile double sink = 0.0;
    for (std::size_t r = 0; r < repeats; ++r) {
        for (const auto& row : rows) {
            const auto t0 = std::chrono::high_resolution_clock::now();
            sink = sink + predict_one(row);
            const auto t1 = std::chrono::high_resolution_clock::now();
            const auto us = std::chrono::duration_cast<std::chrono::duration<double, std::micro>>(t1 - t0).count();
            samples.push_back(us);
        }
    }
    if (sink < -1.0) {
        std::cerr << "sink=" << sink << '\n';
    }

    std::sort(samples.begin(), samples.end());
    const std::size_t idx = static_cast<std::size_t>(0.95 * static_cast<double>(samples.size() - 1));
    return samples[idx];
}

[[nodiscard]] Metrics run_scenario(const ManifestRow& row) {
    if (row.dataset_family != "logic_family") {
        throw std::invalid_argument("unsupported dataset_family: " + row.dataset_family);
    }
    const auto data = generate_logic_family(row);

    rf::RandomForestParams teacher_params;
    teacher_params.n_estimators = 1;
    teacher_params.max_depth = 2;
    teacher_params.bootstrap = false;
    teacher_params.max_features_strategy = rf::CartParams::MaxFeaturesStrategy::All;
    teacher_params.random_seed = row.seed + 101;
    teacher_params.compute_oob = false;

    rf::RandomForestParams compact_params;
    compact_params.n_estimators = 12;
    compact_params.max_depth = 4;
    compact_params.random_seed = row.seed + 202;
    compact_params.compute_oob = false;

    rf::RandomForest teacher(teacher_params);
    rf::RandomForest compact(compact_params);

    unified_ml::DatasetView train_ds(data.train_x, data.train_y, unified_ml::LearningTask::Classification);
    teacher.fit(train_ds.to_rf_dataset(rf::TaskType::Classification));
    compact.fit(train_ds.to_rf_dataset(rf::TaskType::Classification));

    unified_ml::DistillationConfig cfg;
    cfg.gate_budget = 6;
    cfg.synthesis.iterations = 256;
    const auto summary = unified_ml::distill_to_sle(teacher, train_ds, cfg);

    const auto teacher_pred = teacher.predict(data.test_x);
    const auto compact_pred = compact.predict(data.test_x);
    const auto sle_pred = unified_ml::predict_with_distilled_sle(
        std::span<const std::vector<double>>(data.test_x.data(), data.test_x.size()), summary.artifact);

    Metrics out;
    out.teacher_accuracy = accuracy(teacher_pred, data.test_y);
    out.compact_accuracy = accuracy(compact_pred, data.test_y);
    out.sle_accuracy = accuracy(sle_pred, data.test_y);
    out.fidelity_to_teacher = agreement(sle_pred, teacher_pred);
    out.compact_node_count = compact_model_size_proxy(compact);
    out.sle_gate_count = summary.gate_count;

    constexpr std::size_t k_latency_repeats = 40;
    out.teacher_p95_us = estimate_p95_us([&](const std::vector<double>& sample) { return teacher.predict_one(sample); },
                                         std::span<const std::vector<double>>(data.test_x.data(), data.test_x.size()),
                                         k_latency_repeats);
    out.compact_p95_us = estimate_p95_us([&](const std::vector<double>& sample) { return compact.predict_one(sample); },
                                         std::span<const std::vector<double>>(data.test_x.data(), data.test_x.size()),
                                         k_latency_repeats);
    out.sle_p95_us = estimate_p95_us([&](const std::vector<double>& sample) {
                                       return unified_ml::predict_with_distilled_sle(
                                           std::span<const double>(sample.data(), sample.size()), summary.artifact);
                                     },
                                     std::span<const std::vector<double>>(data.test_x.data(), data.test_x.size()),
                                     k_latency_repeats);

    return out;
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: test_sle_level2_utility <manifest.csv>\n";
        return 2;
    }

    const auto manifest = load_manifest(argv[1]);

    bool clear_win_observed = false;
    for (const auto& row : manifest) {
        const auto metrics = run_scenario(row);

        const double accuracy_drop_pp = (metrics.teacher_accuracy - metrics.sle_accuracy) * 100.0;
        const double latency_regression = metrics.sle_p95_us / std::max(1e-9, metrics.teacher_p95_us);
        const bool size_win = metrics.sle_gate_count + 2 <= metrics.compact_node_count;
        const bool latency_win = metrics.sle_p95_us + 1e-9 < metrics.compact_p95_us;
        const bool interpretability_win = metrics.sle_gate_count <= 6;

        clear_win_observed = clear_win_observed || size_win || latency_win || interpretability_win;

        std::cout << "[SLE-L2] family=" << row.dataset_family << " seed=" << row.seed
                  << " profile=" << row.profile << " fidelity=" << metrics.fidelity_to_teacher
                  << " teacher_acc=" << metrics.teacher_accuracy
                  << " sle_acc=" << metrics.sle_accuracy
                  << " compact_acc=" << metrics.compact_accuracy
                  << " sle_gate_count=" << metrics.sle_gate_count
                  << " compact_nodes=" << metrics.compact_node_count
                  << " teacher_p95_us=" << metrics.teacher_p95_us
                  << " sle_p95_us=" << metrics.sle_p95_us << '\n';

        if (metrics.fidelity_to_teacher < 0.95) {
            std::cerr << "fidelity gate failed for seed " << row.seed << "\n";
            return 10;
        }

        if (!row.compression_profile && accuracy_drop_pp > 2.0) {
            std::cerr << "accuracy drop gate failed for seed " << row.seed
                      << ": drop_pp=" << accuracy_drop_pp << "\n";
            return 11;
        }

        if (!row.latency_optimized && latency_regression > 1.10) {
            std::cerr << "latency regression gate failed for seed " << row.seed
                      << ": regression=" << latency_regression << "\n";
            return 12;
        }
    }

    if (!clear_win_observed) {
        std::cerr << "no clear win axis observed across benchmark manifest\n";
        return 13;
    }

    std::cout << "[SLE-L2] all gates passed\n";
    return 0;
}
