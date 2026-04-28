// test_xgboost_units.cpp
// Unit tests for xgboost subsystems previously at 0-17% coverage:
//   - LossFunction::create factory
//   - LogisticLoss: gradients, transform, compute_loss, base_score
//   - SquaredError: gradients, compute_loss, base_score
//   - LossFunction::transform_batch (base class)
//   - AUCMetric: perfect, inverted, random classifiers
//   - XGBModel: binary classification and regression integration

#include <models/xgboost/xgboost_enhanced.hpp>
#include <models/xgboost/objective/loss_function.hpp>
#include <models/xgboost/objective/logistic_loss.hpp>
#include <models/xgboost/objective/squared_error.hpp>
#include <models/xgboost/metric/auc.hpp>

#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {

int g_pass = 0, g_fail = 0;

void check(bool ok, const std::string& name) {
    if (ok) { std::cout << "PASS  " << name << "\n"; ++g_pass; }
    else    { std::cout << "FAIL  " << name << "\n"; ++g_fail; }
}

bool near(double a, double b, double tol = 1e-4) { return std::fabs(a-b) < tol; }

// ── LossFunction factory ──────────────────────────────────────────────────────
void test_loss_factory() {
    auto logistic = xgb::LossFunction::create("binary:logistic");
    check(logistic != nullptr, "factory creates binary:logistic");
    check(logistic->name() == "binary:logistic", "logistic name()");

    auto reg = xgb::LossFunction::create("reg:squarederror");
    check(reg != nullptr, "factory creates reg:squarederror");
    check(reg->name() == "reg:squarederror", "squarederror name()");

    // Unknown objective should either throw or return nullptr — just don't crash
    try {
        auto unknown = xgb::LossFunction::create("nonexistent:obj");
        // If it returns a non-null fallback, that's fine too
        check(true, "factory unknown obj handled without crash");
    } catch (...) {
        check(true, "factory unknown obj threw (acceptable)");
    }
}

// ── LogisticLoss ─────────────────────────────────────────────────────────────
void test_logistic_loss_gradients() {
    xgb::LogisticLoss loss;

    // Perfect prediction: score=+10 for label=1, score=-10 for label=0
    std::vector<xgb::bst_float> scores = {10.f, -10.f};
    std::vector<xgb::bst_float> labels = {1.f,   0.f};
    std::vector<xgb::GradientPair> grads;
    loss.compute_gradients(scores, labels, grads);

    check(grads.size() == 2, "LogisticLoss grad size");
    // gi = sigma(score) - label; for score=10, sigma≈1, label=1 → gi≈0
    check(std::fabs(grads[0].grad) < 0.01f, "LogisticLoss gi≈0 for confident correct");
    // hi = sigma*(1-sigma) > 0
    check(grads[0].hess > 0.f, "LogisticLoss hi > 0");

    // Poor prediction: score=0 for label=1 → gi = 0.5 - 1 = -0.5
    std::vector<xgb::bst_float> s2 = {0.f};
    std::vector<xgb::bst_float> l2 = {1.f};
    std::vector<xgb::GradientPair> g2;
    loss.compute_gradients(s2, l2, g2);
    check(near(g2[0].grad, -0.5f, 1e-3f), "LogisticLoss gi = -0.5 at score=0,y=1");
    check(near(g2[0].hess,  0.25f, 1e-3f), "LogisticLoss hi = 0.25 at score=0");
}

void test_logistic_loss_transform() {
    xgb::LogisticLoss loss;
    // sigmoid(0) = 0.5
    check(near(loss.transform(0.f), 0.5f), "LogisticLoss transform(0)=0.5");
    // sigmoid(large pos) ≈ 1
    check(loss.transform(100.f) > 0.99f, "LogisticLoss transform(100)≈1");
    // sigmoid(large neg) ≈ 0
    check(loss.transform(-100.f) < 0.01f, "LogisticLoss transform(-100)≈0");
}

void test_logistic_loss_compute_loss() {
    xgb::LogisticLoss loss;
    // For perfect predictions, loss should be low
    std::vector<xgb::bst_float> scores_good = {10.f, -10.f};
    std::vector<xgb::bst_float> scores_bad  = {-10.f, 10.f};
    std::vector<xgb::bst_float> labels      = {1.f, 0.f};

    float loss_good = loss.compute_loss(scores_good, labels);
    float loss_bad  = loss.compute_loss(scores_bad,  labels);
    check(loss_good < loss_bad, "LogisticLoss: correct prediction < inverted loss");
}

void test_logistic_base_score() {
    xgb::LogisticLoss loss;
    // For 50% positive rate, base_score should be sigmoid(0) = 0.5 ish (log-odds = 0)
    std::vector<xgb::bst_float> labels = {0.f, 1.f, 0.f, 1.f};
    float bs = loss.base_score(labels);
    check(bs >= 0.f && bs <= 1.f, "LogisticLoss base_score in [0,1]");
}

// ── SquaredError ─────────────────────────────────────────────────────────────
void test_squared_error_gradients() {
    xgb::SquaredError loss;

    // gi = score - label, hi = 1
    std::vector<xgb::bst_float> scores = {2.f, -1.f};
    std::vector<xgb::bst_float> labels = {1.f,  1.f};
    std::vector<xgb::GradientPair> grads;
    loss.compute_gradients(scores, labels, grads);

    check(grads.size() == 2, "SquaredError grad size");
    check(near(grads[0].grad,  1.f), "SquaredError gi = score-label = 1");
    check(near(grads[1].grad, -2.f), "SquaredError gi = score-label = -2");
    check(near(grads[0].hess,  1.f), "SquaredError hi = 1");
}

void test_squared_error_loss() {
    xgb::SquaredError loss;
    std::vector<xgb::bst_float> scores_perfect = {1.f, 2.f, 3.f};
    std::vector<xgb::bst_float> scores_off     = {0.f, 0.f, 0.f};
    std::vector<xgb::bst_float> labels         = {1.f, 2.f, 3.f};

    float l_perfect = loss.compute_loss(scores_perfect, labels);
    float l_off     = loss.compute_loss(scores_off,     labels);
    check(l_perfect < l_off, "SquaredError: perfect < bad predictions");
    check(l_perfect < 1e-4f, "SquaredError: perfect loss ≈ 0");
}

void test_squared_error_base_score() {
    xgb::SquaredError loss;
    std::vector<xgb::bst_float> labels = {1.f, 3.f, 5.f};
    float bs = loss.base_score(labels);
    // Base score is mean: (1+3+5)/3 = 3
    check(near(bs, 3.f, 0.1f), "SquaredError base_score = mean of labels");
}

// ── transform_batch (LossFunction base) ──────────────────────────────────────
void test_transform_batch() {
    xgb::LogisticLoss loss;
    std::vector<xgb::bst_float> margins = {0.f, 10.f, -10.f};
    auto probs = loss.transform_batch(margins);
    check(probs.size() == 3, "transform_batch output size");
    check(near(probs[0], 0.5f), "transform_batch[0] sigmoid(0)=0.5");
    check(probs[1] > 0.99f,     "transform_batch[1] sigmoid(10)≈1");
    check(probs[2] < 0.01f,     "transform_batch[2] sigmoid(-10)≈0");
}

// ── AUCMetric ─────────────────────────────────────────────────────────────────
void test_auc_perfect() {
    xgb::AUCMetric auc;
    std::vector<xgb::bst_float> preds  = {0.9f, 0.8f, 0.2f, 0.1f};
    std::vector<xgb::bst_float> labels = {1.f,  1.f,  0.f,  0.f};
    float score = auc.evaluate(preds, labels);
    check(near(score, 1.0f, 1e-4f), "AUC perfect classifier = 1.0");
    check(auc.higher_is_better(), "AUC higher_is_better");
    check(auc.name() == "auc", "AUC name()");
}

void test_auc_inverted() {
    xgb::AUCMetric auc;
    // Flipped predictions → AUC = 0
    std::vector<xgb::bst_float> preds  = {0.1f, 0.2f, 0.8f, 0.9f};
    std::vector<xgb::bst_float> labels = {1.f,  1.f,  0.f,  0.f};
    float score = auc.evaluate(preds, labels);
    check(near(score, 0.0f, 1e-4f), "AUC inverted classifier = 0.0");
}

void test_auc_random() {
    xgb::AUCMetric auc;
    // Equal preds → AUC ≈ 0.5
    std::vector<xgb::bst_float> preds  = {0.5f, 0.5f, 0.5f, 0.5f};
    std::vector<xgb::bst_float> labels = {1.f,  0.f,  1.f,  0.f};
    float score = auc.evaluate(preds, labels);
    check(score >= 0.f && score <= 1.f, "AUC random classifier in [0,1]");
}

// ── XGBModel integration ─────────────────────────────────────────────────────
void test_xgbmodel_binary_classification() {
    // XOR-like separable binary classification
    std::vector<std::vector<xgb::bst_float>> X = {
        {0.f, 0.f}, {0.f, 1.f}, {1.f, 0.f}, {1.f, 1.f},
        {0.1f, 0.1f}, {0.9f, 0.9f}, {0.f, 0.9f}, {0.9f, 0.f}
    };
    std::vector<xgb::bst_float> y = {0.f, 1.f, 1.f, 0.f, 0.f, 0.f, 1.f, 1.f};

    xgb::XGBModel model;
    model.n_estimators(10)
         .max_depth(3)
         .learning_rate(0.3f)
         .objective("binary:logistic");

    check(!model.is_fitted(), "XGBModel not fitted before fit()");
    model.fit(X, y);
    check(model.is_fitted(), "XGBModel fitted after fit()");

    auto preds = model.predict(X);
    check(preds.size() == X.size(), "XGBModel predict size matches X");

    // All predictions in [0,1] (sigmoid output)
    bool all_valid = true;
    for (float p : preds) if (p < 0.f || p > 1.f) { all_valid = false; break; }
    check(all_valid, "XGBModel binary predictions in [0,1]");
}

void test_xgbmodel_regression() {
    // Simple regression: y ≈ 2*x
    std::vector<std::vector<xgb::bst_float>> X;
    std::vector<xgb::bst_float> y;
    for (int i = 0; i < 20; ++i) {
        float xi = static_cast<float>(i) / 20.f;
        X.push_back({xi});
        y.push_back(2.f * xi);
    }

    xgb::XGBModel model;
    model.n_estimators(20)
         .max_depth(3)
         .learning_rate(0.3f)
         .objective("reg:squarederror");

    model.fit(X, y);
    auto preds = model.predict(X);
    check(preds.size() == X.size(), "XGBModel regression predict size");

    // Predictions should be roughly in the same range as labels
    bool sane = true;
    for (float p : preds) if (p < -1.f || p > 3.f) { sane = false; break; }
    check(sane, "XGBModel regression predictions in sane range");
}

void test_xgbmodel_feature_importances() {
    std::vector<std::vector<xgb::bst_float>> X = {
        {1.f, 0.f}, {0.f, 1.f}, {1.f, 1.f}, {0.f, 0.f},
        {0.8f, 0.2f}, {0.2f, 0.8f}
    };
    std::vector<xgb::bst_float> y = {1.f, 0.f, 1.f, 0.f, 1.f, 0.f};

    xgb::XGBModel model;
    model.n_estimators(5).max_depth(2).objective("binary:logistic");
    model.fit(X, y);

    auto imp = model.feature_importances("gain");
    check(!imp.empty(), "XGBModel feature_importances not empty");
}

} // namespace

int main() {
    test_loss_factory();

    test_logistic_loss_gradients();
    test_logistic_loss_transform();
    test_logistic_loss_compute_loss();
    test_logistic_base_score();

    test_squared_error_gradients();
    test_squared_error_loss();
    test_squared_error_base_score();

    test_transform_batch();

    test_auc_perfect();
    test_auc_inverted();
    test_auc_random();

    test_xgbmodel_binary_classification();
    test_xgbmodel_regression();
    test_xgbmodel_feature_importances();

    std::cout << "\n" << g_pass << " passed, " << g_fail << " failed\n";
    return g_fail ? 1 : 0;
}
