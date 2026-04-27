/**
 * @file main.cpp
 * @brief Minimal production-style SDK integration example.
 *
 * The example intentionally uses only installed public headers and exported
 * targets so it doubles as a packaging smoke test.
 */
#include <unified_ml>

#include <cassert>
#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

static void section(const char* name) {
  std::cout << "\n  " << name << "  \n";
}
static void ok(const char* msg) {
  std::cout << "  [✓] " << msg << "\n";
}

int main() {
  std::cout << "unified_ml v" UNIFIED_ML_VERSION_STRING " — smoke test\n";

  // 1. Version macros          
  section("Version");
  static_assert(UNIFIED_ML_VERSION >= UNIFIED_ML_MAKE_VERSION(1, 0, 0),
      "unified_ml >= 1.0.0 required");
  ok("Version macro check");

  // 2. PCA
  section("PCA");
  {
    core::Matrix X = core::Matrix::from_rows({
      core::Vector({2.5, 2.4, 0.0}),
      core::Vector({0.5, 0.7, 0.1}),
      core::Vector({2.2, 2.9, 0.2}),
      core::Vector({1.9, 2.2, 0.0}),
      core::Vector({3.1, 3.0, 0.1}),
      core::Vector({2.3, 2.7, 0.0})
    });
    pca::PCA pca(2);
    pca.fit(X);
    auto Xt = pca.transform(X);
    assert(Xt.rows() == 6 && Xt.cols() == 2);
    ok("PCA fit + transform (6×3 → 6×2)");
  }

  // 3. Autograd
  section("Autograd");
  {
    autograd::Tensor a({3.0}, {1}, true);
    autograd::Tensor b({2.0}, {1}, true);
    auto c = a * b;   // c = a * b = 6
    c.backward();
    // dc/da = b = 2,  dc/db = a = 3
    assert(std::abs(a.grad()[0] - 2.0) < 1e-6);
    assert(std::abs(b.grad()[0] - 3.0) < 1e-6);
    ok("Autograd backward (a*b, dc/da=2, dc/db=3)");
  }

  // 4. MLP
  section("MLP");
  {
    mlp::Sequential net;
    net.add(std::make_unique<mlp::Linear>(2, 8));
    net.add(std::make_unique<mlp::ReLU>());
    net.add(std::make_unique<mlp::Linear>(8, 1));

    autograd::Tensor x({0.5, -0.3}, {2}, false);
    auto y = net.forward(x);
    assert(y.data().size() == 1);
    ok("MLP forward pass (2→8→1)");
  }

  // 5. Random Forest
  section("Random Forest");
  {
    std::vector<std::vector<double>> X = {
    {0,0},{0,1},{1,0},{1,1},
    {0,0},{0,1},{1,0},{1,1}
    };
    std::vector<double> y = {0,1,1,0,0,1,1,0};
    rf::Dataset dataset(X, y, rf::TaskType::Classification);

    rf::RandomForestParams params;
    params.n_estimators = 10;
    params.max_depth  = 4;
    params.random_seed  = 42;

    rf::RandomForest forest(params);
    forest.fit(dataset);
    auto pred = forest.predict(dataset);
    assert(pred.size() == 8);
    ok("Random Forest fit + predict (8 samples)");
  }

  // 6. DBSCAN
  section("DBSCAN");
  {
    std::vector<dbscan::Point> pts = {
    {1.0,1.0},{1.1,1.0},{1.0,1.1},{1.05,1.05},
    {5.0,5.0},{5.1,5.0},{5.0,5.1},
    {9.0,0.0} // noise
    };
    dbscan::DBSCAN db(0.5, 3);
    auto labels = db.fit(pts);
    assert(labels.size() == pts.size());
    assert(labels.back() == -1);  // last point is noise
    ok("DBSCAN fit (2 clusters + 1 noise)");
  }

  // 7. Isolation Forest
  section("Isolation Forest");
  {
    std::vector<std::vector<double>> X(50, std::vector<double>(2));
    for (int i = 0; i < 50; ++i)
    X[i] = {static_cast<double>(i)/50.0, static_cast<double>(i%10)/10.0};

    iforest::IsolationForest iso(50, 256, 42);
    iso.fit(X);
    // score() is the correct method name (not anomaly_score)
    double s = iso.score(X[0]);
    assert(s >= 0.0 && s <= 1.0);
    ok("IsolationForest fit + score");
  }

  std::cout << "\n✓ All smoke tests passed.\n";
  return 0;
}
