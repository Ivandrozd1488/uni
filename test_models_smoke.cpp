#include <autograd/autograd.h>
#include <core/linalg.hpp>
#include <core/quantization.hpp>
#include <models/dbscan/dbscan.hpp>
#include <models/deep_onet/model.hpp>
#include <models/iforest/isolation_forest.hpp>
#include <models/kriging/kriging.hpp>
#include <models/mlp/activation.hpp>
#include <models/mlp/loss.hpp>
#include <models/mlp/linear.hpp>
#include <models/mlp/model.hpp>
#include <models/mlp/sequential.hpp>
#include <models/pca/pca.hpp>
#include <models/pideeponet/pideeponet.hpp>
#include <models/pinn/neural_network.hpp>
#include <models/rf/dataset.hpp>
#include <models/rf/random_forest.hpp>
#include <models/sindy/sindy.hpp>
#include <models/svm/svm.hpp>
#include <models/transformer/transformer_block.hpp>
#include <models/xgboost/xgboost_enhanced.hpp>
#include <models/gp/gaussian_process.hpp>

#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int g_pass = 0;
int g_fail = 0;

void check(bool ok, const std::string& name) {
    if (ok) {
        std::cout << "PASS  " << name << "\n";
        ++g_pass;
    } else {
        std::cout << "FAIL  " << name << "\n";
        ++g_fail;
    }
}

void test_rf() {
    std::vector<std::vector<double>> X{{0, 0}, {0, 1}, {1, 0}, {1, 1}, {0.9, 0.8}, {0.1, 0.2}};
    std::vector<double> y{0, 1, 1, 0, 0, 0};
    rf::Dataset ds(X, y, rf::TaskType::Classification);

    rf::RandomForestParams params;
    params.n_estimators = 8;
    params.max_depth = 4;
    params.compute_oob = false;

    rf::RandomForest model(params);
    model.fit(ds);
    auto pred = model.predict(ds);

    check(pred.size() == X.size(), "rf.predict size matches");
    check(model.n_trees() == static_cast<std::size_t>(params.n_estimators), "rf.n_trees matches n_estimators");
}

void test_pca() {
    core::Matrix X(5, 3, 0.0);
    X(0, 0) = 1.0; X(0, 1) = 2.0; X(0, 2) = 3.0;
    X(1, 0) = 2.0; X(1, 1) = 3.0; X(1, 2) = 4.0;
    X(2, 0) = 3.0; X(2, 1) = 4.0; X(2, 2) = 5.0;
    X(3, 0) = 4.0; X(3, 1) = 5.0; X(3, 2) = 6.0;
    X(4, 0) = 5.0; X(4, 1) = 6.0; X(4, 2) = 7.0;

    pca::PCA model(2);
    auto Z = model.fit_transform(X);
    check(model.is_fitted(), "pca fitted");
    check(Z.rows() == 5 && Z.cols() == 2, "pca output shape 5x2");
}

void test_dbscan() {
    std::vector<dbscan::Point> pts{{0.0, 0.0}, {0.1, 0.1}, {0.2, 0.2}, {3.0, 3.0}, {3.1, 3.2}, {10.0, 10.0}};
    dbscan::DBSCAN model(0.35, 2);
    auto labels = model.fit(pts);
    check(labels.size() == pts.size(), "dbscan labels size matches");
    check(model.numClusters() >= 1, "dbscan found at least one cluster");
}

void test_iforest() {
    std::vector<std::vector<double>> X{{0.0, 0.0}, {0.1, 0.1}, {0.2, 0.2}, {0.3, 0.3}, {8.0, 8.0}};
    iforest::IsolationForest model(16, 4, -1, 42);
    model.fit(X);
    auto scores = model.score_batch(X);
    check(scores.size() == X.size(), "iforest score_batch size matches");
    check(scores.back() > scores.front(), "iforest anomaly score distinguishes outlier");
}

void test_mlp() {
    auto seq = std::make_unique<mlp::Sequential>();
    seq->add(std::make_unique<mlp::Linear>(2, 4));
    seq->add(std::make_unique<mlp::Tanh>());
    seq->add(std::make_unique<mlp::Linear>(4, 1));

    mlp::Model model(std::move(seq), 1e-2);
    autograd::Tensor x({0.0, 1.0}, {1, 2}, false);
    autograd::Tensor y({1.0}, {1, 1}, false);

    auto loss = model.train_step(x, y, mlp::mse_loss);
    check(loss.numel() == 1, "mlp train_step scalar loss");

    auto seq_auto = std::make_unique<mlp::Sequential>();
    seq_auto->add(std::make_unique<mlp::Linear>(2, 8));
    seq_auto->add(std::make_unique<mlp::Tanh>());
    seq_auto->add(std::make_unique<mlp::Linear>(8, 1));
    mlp::Model auto_model(std::move(seq_auto), 1e-3, mlp::OptimizerType::Auto);
    check(auto_model.selected_optimizer_type() == mlp::OptimizerType::NAdam,
          "mlp auto optimizer picks NAdam for small network");
}

void test_pinn() {
    pinn::NeuralNetwork net({1, 8, 1}, pinn::Activation::Tanh, 42);
    auto d = net.forward_derivs_1d(0.25);
    check(d.u.numel() == 1, "pinn forward_derivs_1d returns scalar u");
    check(d.du_dx.numel() == 1 && d.d2u_dx2.numel() == 1, "pinn derivatives scalar shapes");
}

void test_deeponet() {
    deep_onet::DeepONet model(/*branch_in=*/2, {4}, /*trunk_in=*/2, {4}, /*latent=*/3,
                              deep_onet::Activation::Tanh, deep_onet::Activation::Tanh);

    autograd::Tensor u_batch({1.0, 0.5, 0.2, 1.2}, {2, 2}, false);
    autograd::Tensor y_batch({0.1, 0.2, 0.3, 0.4}, {2, 2}, false);

    auto out = model.forward(u_batch, y_batch);
    check(out.shape().size() == 2 && out.shape()[0] == 2 && out.shape()[1] == 1,
          "deeponet forward output shape [batch,1]");
}

void test_transformer() {
    transformer::TransformerBlock block(8, 16, 2, true);
    autograd::Tensor x(std::vector<double>(2 * 3 * 8, 0.1), {2, 3, 8}, false);
    autograd::Tensor mask({1.0, 1.0, 0.0, 1.0, 1.0, 1.0}, {2, 3}, false);
    auto out = block.forward(x, &mask);
    auto attn = block.attention_weights(x, &mask);
    check(out.shape() == std::vector<std::size_t>({2, 3, 8}), "transformer block output shape [2,3,8]");
    check(attn.shape() == std::vector<std::size_t>({2, 2, 3, 3}), "transformer attention map shape [2,2,3,3]");

    transformer::TransformerConfig cfg;
    cfg.embed_dim = 8;
    cfg.ff_hidden_dim = 16;
    cfg.num_heads = 2;
    cfg.num_layers = 3;
    cfg.vocab_size = 10;
    cfg.max_seq_len = 8;
    cfg.causal = true;
    transformer::TransformerEncoder encoder(cfg);
    cfg.num_classes = 4;
    transformer::TransformerEncoder clf_encoder(cfg);
    auto enc = clf_encoder.forward_tokens({{1, 2, 3}, {3, 2, 1}});
    auto maps = clf_encoder.collect_attention_maps(x, &mask);
    auto logits = clf_encoder.classify_tokens({{1, 2, 3}, {3, 2, 1}});
    auto lm_logits = encoder.language_model_logits_tokens({{1, 2, 3}, {3, 2, 1}});
    auto lm_loss = encoder.next_token_loss({{1, 2, 3}, {3, 2, 1}});
    auto decoded = encoder.greedy_decode({1, 2, 3}, 2);
    auto topk_decoded = encoder.top_k_decode({1, 2, 3}, 2, 3);
    auto beam_decoded = encoder.beam_search_decode({1, 2, 3}, 2, 3);
    auto incremental_decoded = encoder.incremental_decode({1, 2, 3}, 2);
    transformer::TransformerSamplingConfig sampling_cfg;
    sampling_cfg.top_k = 3;
    sampling_cfg.temperature = 0.9;
    sampling_cfg.seed = 7;
    sampling_cfg.eos_token = 9;
    auto sampled_decoded = encoder.sample_decode({1, 2, 3}, 2, sampling_cfg);
    auto batch_decoded = encoder.batch_generate({{1, 2, 3}, {2, 3, 4}}, 2, &sampling_cfg, 0, false);
    transformer::TransformerSeq2Seq seq2seq(cfg);
    autograd::Tensor source_mask(std::vector<double>{1.0, 1.0, 0.0}, {1, 3}, false);
    autograd::Tensor target_mask(std::vector<double>{1.0, 0.0}, {1, 2}, false);
    std::vector<double> source_mask_vec{1.0, 1.0, 0.0};
    auto seq2seq_logits = seq2seq.forward_tokens({{1, 2, 3}}, {{4, 5}}, &source_mask, &target_mask);
    auto seq2seq_loss = seq2seq.teacher_forcing_loss({{1, 2, 3}}, {{4, 5, 6}});
    auto cross_maps = seq2seq.collect_cross_attention_maps({{1, 2, 3}}, {{4, 5}});
    auto seq2seq_decoded = seq2seq.greedy_decode({1, 2, 3}, {4}, 2, &source_mask_vec);
    auto seq2seq_topk = seq2seq.top_k_decode({1, 2, 3}, {4}, 2, 3, &source_mask_vec);
    auto seq2seq_beam = seq2seq.beam_search_decode({1, 2, 3}, {4}, 2, 3, &source_mask_vec);
    auto seq2seq_incremental = seq2seq.incremental_decode({1, 2, 3}, {4}, 2, &source_mask_vec);
    auto seq2seq_sampled = seq2seq.sample_decode({1, 2, 3}, {4}, 2, sampling_cfg, &source_mask_vec);
    transformer::TransformerSystem system(cfg);
    auto system_enc = system.encode_tokens({{1, 2, 3}});
    auto system_cls = system.classify_tokens({{1, 2, 3}});
    auto system_lm = system.language_model_logits_tokens({{1, 2, 3}});
    auto system_s2s = system.seq2seq_logits({{1, 2, 3}}, {{4, 5}}, &source_mask, &target_mask);
    auto system_gen = system.generate_causal({1, 2, 3}, 2, &sampling_cfg, 0, false);
    auto system_seq_gen = system.generate_seq2seq({1, 2, 3}, {4}, 2, &sampling_cfg, 0, false, &source_mask_vec);
    auto system_causal_batch = system.generate_causal_batch({{1, 2, 3}, {2, 3, 4}}, 2, &sampling_cfg, 0, false);
    std::vector<std::vector<double>> source_mask_batch{{1.0, 1.0, 0.0}, {1.0, 0.0, 0.0}};
    auto system_seq_batch = system.generate_seq2seq_batch({{1, 2, 3}, {2, 3, 4}}, {{4}, {5}}, 2, &sampling_cfg, 0, false, &source_mask_batch);
    auto system_maps = system.seq2seq_cross_attention_maps({{1, 2, 3}}, {{4, 5}});
    auto decoder_step = block.decoder_step(x, x, &mask, &mask);
    auto cross = block.cross_attention(x, x, &mask);
    check(enc.shape() == std::vector<std::size_t>({2, 3, 8}), "transformer encoder token forward shape [2,3,8]");
    check(maps.size() == 3, "transformer encoder returns per-layer attention maps");
    check(logits.shape() == std::vector<std::size_t>({2, 4}), "transformer classifier head logits shape [2,4]");
    check(lm_logits.shape() == std::vector<std::size_t>({2, 3, 10}), "transformer LM logits shape [2,3,10]");
    check(std::isfinite(lm_loss) && lm_loss >= 0.0, "transformer next-token loss finite");
    check(decoded.size() >= 4, "transformer greedy decode extends prompt");
    check(topk_decoded.size() >= 4, "transformer top-k decode extends prompt");
    check(beam_decoded.size() >= 4, "transformer beam decode extends prompt");
    check(incremental_decoded.size() >= 4, "transformer incremental decode extends prompt");
    check(sampled_decoded.size() >= 4 || sampled_decoded.back() == sampling_cfg.eos_token, "transformer sampled decode extends prompt or stops on eos");
    check(batch_decoded.size() == 2, "transformer batch generate returns batch");
    check(seq2seq_logits.shape() == std::vector<std::size_t>({1, 2, 10}), "transformer seq2seq logits shape [1,2,10]");
    check(std::isfinite(seq2seq_loss) && seq2seq_loss >= 0.0, "transformer seq2seq teacher forcing loss finite");
    check(cross_maps.size() == 3, "transformer seq2seq cross attention map count");
    check(seq2seq_decoded.size() >= 2, "transformer seq2seq greedy decode extends prompt");
    check(seq2seq_topk.size() >= 2, "transformer seq2seq top-k decode extends prompt");
    check(seq2seq_beam.size() >= 2, "transformer seq2seq beam decode extends prompt");
    check(seq2seq_incremental.size() >= 2, "transformer seq2seq incremental decode extends prompt");
    check(seq2seq_sampled.size() >= 2, "transformer seq2seq sampled decode extends prompt");
    check(system_enc.shape() == std::vector<std::size_t>({1, 3, 8}), "transformer system encode shape [1,3,8]");
    check(system_cls.shape() == std::vector<std::size_t>({1, 4}), "transformer system classify shape [1,4]");
    check(system_lm.shape() == std::vector<std::size_t>({1, 3, 10}), "transformer system lm shape [1,3,10]");
    check(system_s2s.shape() == std::vector<std::size_t>({1, 2, 10}), "transformer system seq2seq shape [1,2,10]");
    check(system_gen.size() >= 4, "transformer system causal generation extends prompt");
    check(system_seq_gen.size() >= 2, "transformer system seq2seq generation extends prompt");
    check(system_causal_batch.size() == 2, "transformer system causal batch generation size");
    check(system_seq_batch.size() == 2, "transformer system seq2seq batch generation size");
    check(system_maps.size() == 3, "transformer system cross attention map count");
    check(decoder_step.shape() == std::vector<std::size_t>({2, 3, 8}), "transformer decoder step shape [2,3,8]");
    check(cross.shape() == std::vector<std::size_t>({2, 3, 8}), "transformer cross attention shape [2,3,8]");
}

void test_svm() {
    std::vector<std::vector<double>> X{{-1.0, -1.0}, {-0.5, -0.2}, {0.5, 0.7}, {1.0, 1.2}, {2.0, -1.0}, {2.2, -0.8}};
    std::vector<int> y{0, 0, 1, 1, 2, 2};
    svm::SVMParams params;
    params.kernel = svm::KernelType::Polynomial;
    params.multiclass_strategy = svm::MultiClassStrategy::OneVsRest;
    svm::SVM model(params);
    model.fit(X, y);
    auto pred = model.predict(X);
    auto detail = model.predict_multiclass({0.8, 0.9});
    auto batch_proba = model.predict_proba(X);
    auto diag = model.diagnostics(X, y);
    check(model.is_fitted(), "svm fitted");
    check(model.n_classes() == 3, "svm multiclass class count");
    check(pred.size() == X.size(), "svm predict size matches");
    check(detail.probabilities.size() == 3, "svm multiclass probabilities size");
    check(batch_proba.size() == X.size() && batch_proba[0].size() == 3, "svm batch probability shape");
    check(diag.training_accuracy >= 0.6, "svm diagnostics training accuracy sane");

    svm::SVMParams svr_params;
    svr_params.mode = svm::SVMMode::Regression;
    svr_params.kernel = svm::KernelType::RBF;
    svr_params.epsilon = 0.05;
    svr_params.optimization = svm::SVMOptimization::SMOStyle;
    svr_params.variant = svm::SVMVariant::Nu;
    svr_params.nu = 0.4;
    svr_params.quantile_tau = 0.8;
    svm::SVM svr(svr_params);
    svr.fit_regression({{0.0}, {1.0}, {2.0}, {3.0}}, {0.0, 1.0, 2.0, 3.0}, {1.0, 2.0, 0.5, 1.5});
    auto reg_pred = svr.predict_regression(std::vector<double>{1.5});
    auto reg_diag = svr.regression_diagnostics({{0.0}, {1.0}, {2.0}, {3.0}}, {0.0, 1.0, 2.0, 3.0});
    check(std::isfinite(reg_pred), "svm regression prediction finite");
    check(reg_diag.rmse >= 0.0 && reg_diag.mae >= 0.0, "svm regression diagnostics finite");
    check(reg_diag.mean_predictive_stddev >= 0.0 && reg_diag.nominal_coverage_95 >= 0.0 && reg_diag.nominal_coverage_95 <= 1.0,
          "svm regression confidence diagnostics finite");

    svm::SVM weighted_cls(params);
    weighted_cls.fit(X, y, {1.0, 1.0, 1.0, 2.0, 0.5, 0.5});
    auto weighted_pred = weighted_cls.predict_multiclass({0.8, 0.9});
    check(weighted_pred.probabilities.size() == 3, "svm sample-weighted multiclass prediction works");
}

void test_gp() {
    std::vector<std::vector<double>> X{{0.0}, {0.5}, {1.0}, {1.5}, {2.0}};
    std::vector<std::vector<double>> Y{{0.0, 1.0}, {0.5, 0.5}, {1.0, 0.0}, {1.5, -0.5}, {2.0, -1.0}};
    gp::GPParams params;
    params.kernel = gp::KernelType::RBFPlusLinear;
    params.approximation = gp::ApproximationType::SparseNystrom;
    params.inducing_points = 3;
    params.output_correlation = 0.35;
    params.use_inducing_correction = true;
    gp::GaussianProcessRegressor model(params);
    model.set_coregionalization_matrix({{1.0, 0.4}, {0.4, 1.0}});
    model.fit_multi_output(X, Y);
    auto p = model.predict_multi_output(std::vector<double>{0.25});
    auto cov = model.predict_multi_output_covariance({0.25});
    model.append_multi_output_observation({2.5}, {2.5, -1.5}, true);
    check(model.is_fitted(), "gp fitted");
    check(model.output_dim() == 2, "gp multi-output dimension");
    check(model.training_point_count() >= 3, "gp append observation keeps active training set populated");
    check(model.log_marginal_likelihood() < 1e9, "gp log marginal likelihood finite");
    check(p.size() == 2 && std::isfinite(p[0]) && std::isfinite(p[1]), "gp finite multi-output prediction");
    check(cov.size() == 2 && cov[0].size() == 2 && std::isfinite(cov[0][1]), "gp multi-output covariance shape");
    check(cov[0][1] > 0.0, "gp coregionalization induces cross-output covariance");

    gp::GPParams opt_params;
    gp::GaussianProcessRegressor opt_model(opt_params);
    opt_model.optimize_hyperparameters(X, {0.0, 0.5, 1.0, 1.5, 2.0}, {{0.5, 1.0}, {0.5, 1.5}, {1e-4, 1e-3}});
    check(opt_model.is_fitted(), "gp hyperparameter optimization fits model");

    gp::GPParams cls_params;
    cls_params.task = gp::TaskType::BinaryClassification;
    cls_params.kernel = gp::KernelType::RBF;
    gp::GaussianProcessRegressor cls_model(cls_params);
    cls_model.fit({{-1.0}, {-0.5}, {0.5}, {1.0}}, {0.0, 0.0, 1.0, 1.0});
    auto cls_pred = cls_model.predict_one({0.75});
    auto cls_probs = cls_model.predict_class_probabilities({{-0.25}, {0.25}, {0.75}});
    auto ucb = cls_model.select_next_point_ucb({{-0.25}, {0.25}, {0.75}}, 2.0);
    auto ei = cls_model.select_next_point_expected_improvement({{-0.25}, {0.25}, {0.75}}, 1.0, 0.01);
    check(cls_pred.probability >= 0.0 && cls_pred.probability <= 1.0, "gp classification probability bounded");
    check(cls_probs.size() == 3 && cls_probs[0] >= 0.0 && cls_probs[0] <= 1.0, "gp batch classification probabilities bounded");
    check(ucb.index < 3 && std::isfinite(ucb.score), "gp ucb acquisition valid");
    check(ei.index < 3 && std::isfinite(ei.score), "gp expected improvement acquisition valid");
}

void test_sindy() {
    std::vector<std::vector<double>> X{{1.0, 0.0}, {2.0, 0.5}, {3.0, 1.0}, {4.0, 1.5}};
    std::vector<std::vector<double>> Xdot{{2.0, 1.0}, {4.0, 1.5}, {6.0, 2.0}, {8.0, 2.5}};
    sindy::SINDyParams params;
    params.polynomial_order = 3;
    params.integrator = sindy::IntegratorType::RK4;
    params.derivative_mode = sindy::DerivativeMode::SmoothedFiniteDifference;
    params.smoothing_window = 1;
    params.weak_window = 2;
    params.include_trig = true;
    params.include_inverse = true;
    params.library = sindy::FeatureLibraryType::Generalized;
    sindy::SINDy model(params);
    model.fit(X, Xdot);
    auto eqs = model.equations();
    auto pred = model.predict_derivative({{1.5, 0.3}});
    check(model.is_fitted(), "sindy fitted");
    check(model.feature_count() > 6, "sindy rich feature library");
    check(!eqs.empty(), "sindy equations non-empty");
    check(pred.size() == 1 && pred[0].size() == 2, "sindy predict derivative shape");

    sindy::SINDy control_model(params);
    control_model.fit_with_control(X, {{0.1}, {0.2}, {0.3}, {0.4}}, Xdot);
    auto ctrl_pred = control_model.predict_derivative_with_control({{1.5, 0.3}}, {{0.25}});
    auto ctrl_traj = control_model.simulate_with_control({1.0, 0.0}, {{0.1}, {0.1}, {0.1}}, 0.1);
    auto free_traj = model.simulate({1.0, 0.0}, 0.1, 5);
    sindy::SINDy traj_model(params);
    traj_model.fit_from_trajectory(X, 0.1);
    auto weak_params = params;
    weak_params.derivative_mode = sindy::DerivativeMode::WeakIntegral;
    sindy::SINDy weak_model(weak_params);
    weak_model.fit_from_trajectory(X, 0.1);
    auto ensemble_params = weak_params;
    ensemble_params.ensemble_models = 5;
    ensemble_params.ensemble_subsample_ratio = 0.75;
    ensemble_params.stability_threshold = 0.4;
    ensemble_params.bag_trajectories = true;
    ensemble_params.weak_test_functions = 4;
    ensemble_params.bootstrap_samples = 8;
    sindy::SINDy ensemble_model(ensemble_params);
    ensemble_model.fit_ensemble(X, Xdot);
    sindy::SINDy multi_traj_model(ensemble_params);
    multi_traj_model.fit_multi_trajectory({X, X}, 0.1);
    auto stability = ensemble_model.stability_report();
    sindy::SINDy multi_ctrl_model(ensemble_params);
    multi_ctrl_model.fit_multi_trajectory_with_control({X, X}, {{{0.1}, {0.2}, {0.3}, {0.4}}, {{0.2}, {0.2}, {0.2}, {0.2}}}, 0.1);
    check(control_model.has_control(), "sindy control-aware fit enabled");
    check(ctrl_pred.size() == 1 && ctrl_pred[0].size() == 2, "sindy controlled derivative shape");
    check(ctrl_traj.size() == 4 && ctrl_traj.front().size() == 2, "sindy controlled simulation shape");
    check(free_traj.size() == 6 && free_traj.front().size() == 2, "sindy free simulation shape");
    check(traj_model.is_fitted(), "sindy trajectory-based fit works");
    check(weak_model.is_fitted(), "sindy weak-form trajectory fit works");
    check(ensemble_model.is_fitted(), "sindy ensemble stability-selection fit works");
    check(multi_traj_model.is_fitted(), "sindy multi-trajectory fit works");
    check(multi_ctrl_model.is_fitted(), "sindy multi-trajectory controlled fit works");
    check(!stability.feature_names.empty() && !stability.selection_frequency.empty(), "sindy stability report populated");
    check(!stability.ci_lower.empty() && !stability.ci_upper.empty(), "sindy term confidence intervals populated");
    check(!stability.support_path_thresholds.empty() && !stability.support_path_coefficients.empty(), "sindy support path populated");
    check(stability.model_selection_summary.size() == 2, "sindy model selection summary populated");
}

void test_pideeponet() {
    pideeponet::PIDeepONet model(/*branch_in=*/2, {4}, /*trunk_in=*/1, {4}, /*latent=*/3,
                                 deep_onet::Activation::Tanh, deep_onet::Activation::Tanh);
    autograd::Tensor u_batch({1.0, 0.5, 0.2, 1.2}, {2, 2}, false);
    autograd::Tensor y_batch({0.1, 0.2}, {2, 1}, false);
    autograd::Tensor target({0.3, 0.4}, {2, 1}, false);
    auto loss = model.loss(
        u_batch, y_batch, target,
        [](const autograd::Tensor& pred, const autograd::Tensor&) {
            return pred;
        },
        0.5, 1.0);
    check(loss.total.numel() == 1, "pideeponet total loss scalar");
    check(loss.data.numel() == 1 && loss.physics.numel() == 1, "pideeponet loss components scalar");
}

void test_quantization() {
    autograd::Tensor w({0.5, -0.25, 0.75, 1.0}, {2, 2}, false);
    autograd::Tensor b({0.1, -0.2}, {2}, false);
    autograd::Tensor x({1.0, 2.0, -1.0, 0.5}, {2, 2}, false);
    auto qw = core::quantize_per_channel_symmetric(w);
    auto yq = core::quantized_linear(x, qw, b);
    check(yq.shape() == std::vector<std::size_t>({2, 2}), "quantized linear output shape [2,2]");

    auto qp = core::calibrate_quant_params(x);
    auto qx = core::quantize_per_tensor(x, qp);
    auto dx = core::dequantize_per_tensor(qx);
    check(dx.shape() == x.shape(), "dequantized tensor shape matches");
}

void test_xgboost() {
    std::vector<std::vector<xgb::bst_float>> X{{0.f, 0.f}, {0.f, 1.f}, {1.f, 0.f}, {1.f, 1.f}};
    std::vector<xgb::bst_float> y{0.f, 1.f, 1.f, 0.f};

    xgb::XGBModel model("binary");
    model.n_estimators(3).max_depth(2).learning_rate(0.3f).subsample(1.0f).colsample(1.0f).verbose(0);
    model.fit(X, y);

    auto pred = model.predict(X);
    check(pred.size() == X.size(), "xgboost predict size matches");
    check(model.is_fitted(), "xgboost fitted flag");
}

void test_kriging() {
    std::vector<kriging::Point2D> pts{
        {0.0, 0.0},
        {0.0, 1.0},
        {1.0, 0.0},
        {1.0, 1.0}
    };
    std::vector<double> vals{0.0, 1.0, 1.0, 2.0};

    kriging::OrdinaryKriging model(
        kriging::VariogramType::Exponential,
        /*nugget=*/0.0,
        /*sill=*/1.0,
        /*range=*/1.5);
    model.fit(pts, vals);

    auto pred = model.predict_with_variance({0.5, 0.5});
    check(model.is_fitted(), "kriging fitted");
    check(model.n_samples() == pts.size(), "kriging n_samples matches");
    check(std::isfinite(pred.value) && std::isfinite(pred.variance), "kriging finite prediction and variance");
    check(pred.variance >= 0.0, "kriging non-negative variance");
}

} // namespace

int main() {
    try {
        test_rf();
        test_pca();
        test_dbscan();
        test_iforest();
        test_mlp();
        test_pinn();
        test_deeponet();
        test_transformer();
        test_svm();
        test_gp();
        test_sindy();
        test_pideeponet();
        test_quantization();
        test_xgboost();
        test_kriging();
    } catch (const std::exception& e) {
        std::cerr << "EXCEPTION: " << e.what() << "\n";
        return 2;
    }

    std::cout << "\nSUMMARY: PASS=" << g_pass << " FAIL=" << g_fail << "\n";
    return g_fail == 0 ? 0 : 1;
}
