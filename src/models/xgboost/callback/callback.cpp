// ═══════════════════════════════════════════════════════════════════════════
//  src/callback/callback.cpp
//
//  Implementation of the typed callback system.
// ═══════════════════════════════════════════════════════════════════════════

// M_PI is not defined by default on MSVC — define before any math include
#ifndef M_PI
#  define M_PI 3.14159265358979323846
#endif

#include "models/xgboost/callback/callback.hpp"
#include "models/xgboost/booster/trainer.hpp"   // TrainHistory
#include "models/xgboost/booster/gradient_booster.hpp"
#include "models/xgboost/utils/logger.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <regex>
#include <stdexcept>

namespace xgb {

//  Helper: retrieve a metric value from TrainHistory     
static bst_float get_metric(const TrainHistory& history,
           bst_int     round,
           const std::string&  dataset,
           const std::string&  metric)
{
  for (const auto& r : history.results) {
    if (r.round    == round  &&
    r.dataset_name == dataset &&
    r.metric_name  == metric) {
    return r.value;
    }
  }
  // Return sentinel if not found (dataset not in eval set yet)
  return std::numeric_limits<bst_float>::quiet_NaN();
}

// ═══════════════════════════════════════════════════════════════════════════
//  EarlyStopping
// ═══════════════════════════════════════════════════════════════════════════

EarlyStopping::EarlyStopping(std::string dataset_name,
           std::string metric_name,
           bst_int   rounds,
           bool    higher_is_better)
  : dataset_name_(std::move(dataset_name))
  , metric_name_ (std::move(metric_name))
  , rounds_  (rounds)
  , higher_is_better_(higher_is_better)
  , best_score_(higher_is_better
      ? -std::numeric_limits<bst_float>::infinity()
      :  std::numeric_limits<bst_float>::infinity())
{}

void EarlyStopping::on_train_begin(CallbackContext& /*ctx*/)
{
  best_iteration_ = 0;
  no_improve_count_ = 0;
  best_score_ = higher_is_better_
      ? -std::numeric_limits<bst_float>::infinity()
      :  std::numeric_limits<bst_float>::infinity();
}

void EarlyStopping::on_iteration_end(CallbackContext& ctx)
{
  if (!ctx.history) return;

  const bst_float val = get_metric(*ctx.history,
             ctx.current_round,
             dataset_name_,
             metric_name_);

  if (std::isnan(val)) return; // dataset not in eval sets

  if (is_better(val, best_score_)) {
    best_score_   = val;
    best_iteration_ = ctx.current_round;
    no_improve_count_ = 0;
  } else {
    ++no_improve_count_;
  }

  if (no_improve_count_ >= rounds_) {
    Logger::instance().log(
      LogLevel::INFO,
      "Early stopping triggered",
      {
        {"event", "early_stopping"},
        {"round", std::to_string(ctx.current_round)},
        {"best_round", std::to_string(best_iteration_)},
        {"metric", metric_name_},
        {"best_score", std::to_string(best_score_)}
      });
    ctx.stop_training  = true;
    ctx.best_iteration = best_iteration_;
  }
}

void EarlyStopping::on_train_end(CallbackContext& /*ctx*/)
{
  Logger::instance().log(
    LogLevel::INFO,
    "Training finished",
    {
      {"event", "early_stopping_summary"},
      {"best_iteration", std::to_string(best_iteration_)},
      {"metric", metric_name_},
      {"best_score", std::to_string(best_score_)}
    });
}

bool EarlyStopping::is_better(bst_float candidate, bst_float best) const
{
  return higher_is_better_ ? (candidate > best) : (candidate < best);
}

// ═══════════════════════════════════════════════════════════════════════════
//  ModelCheckpoint
// ═══════════════════════════════════════════════════════════════════════════

ModelCheckpoint::ModelCheckpoint(std::string path_template,
           std::string dataset_name,
           std::string metric_name,
           bool    higher_is_better,
           bool    save_best_only)
  : path_template_  (std::move(path_template))
  , dataset_name_   (std::move(dataset_name))
  , metric_name_  (std::move(metric_name))
  , higher_is_better_ (higher_is_better)
  , save_best_only_ (save_best_only)
  , best_score_(higher_is_better
      ? -std::numeric_limits<bst_float>::infinity()
      :  std::numeric_limits<bst_float>::infinity())
{}

void ModelCheckpoint::on_iteration_end(CallbackContext& ctx)
{
  if (!ctx.history || !ctx.booster) return;

  const bst_float val = get_metric(*ctx.history,
             ctx.current_round,
             dataset_name_,
             metric_name_);
  if (std::isnan(val)) return;

  const bool improved = is_better(val, best_score_);
  if (save_best_only_ && !improved) return;

  if (improved) best_score_ = val;

  const std::string path = make_path(ctx.current_round);
  ctx.booster->save_model(path);
  Logger::instance().log(
    LogLevel::INFO,
    "Model checkpoint saved",
    {
      {"event", "model_checkpoint"},
      {"path", path},
      {"metric", metric_name_},
      {"metric_value", std::to_string(val)},
      {"round", std::to_string(ctx.current_round)}
    });
}

std::string ModelCheckpoint::make_path(bst_int round) const
{
  std::string path = path_template_;
  const std::string token = "{round}";
  const auto pos = path.find(token);
  if (pos != std::string::npos) {
    path.replace(pos, token.size(), std::to_string(round));
  }
  return path;
}

bool ModelCheckpoint::is_better(bst_float candidate, bst_float best) const
{
  return higher_is_better_ ? (candidate > best) : (candidate < best);
}

// ═══════════════════════════════════════════════════════════════════════════
//  LearningRateScheduler
// ═══════════════════════════════════════════════════════════════════════════

LearningRateScheduler::ScheduleFn
LearningRateScheduler::cosine_annealing(bst_float eta_max,
               bst_float eta_min,
               bst_int T_max)
{
  return [eta_max, eta_min, T_max](bst_int round) -> bst_float {
    const float t = static_cast<float>(round % T_max);
    const float cos_val = std::cos(2.f * static_cast<float>(M_PI) * t
             / static_cast<float>(T_max));
    return eta_min + 0.5f * (eta_max - eta_min) * (1.f + cos_val);
  };
}

LearningRateScheduler::ScheduleFn
LearningRateScheduler::step_decay(bst_float initial_lr,
             bst_float drop_factor,
             bst_int step_size)
{
  return [initial_lr, drop_factor, step_size](bst_int round) -> bst_float {
    const bst_int steps = round / step_size;
    return initial_lr * std::pow(drop_factor, static_cast<float>(steps));
  };
}

void LearningRateScheduler::on_iteration_begin(CallbackContext& ctx)
{
  if (!ctx.booster) return;
  const bst_float new_lr = fn_(ctx.current_round);
  ctx.booster->mutable_config().eta = new_lr;
  ctx.booster->mutable_config().tree.colsample_bytree =
    ctx.booster->mutable_config().tree.colsample_bytree;  // no-op, keeps consistent
}

// ═══════════════════════════════════════════════════════════════════════════
//  LoggingCallback
// ═══════════════════════════════════════════════════════════════════════════

void LoggingCallback::on_iteration_end(CallbackContext& ctx)
{
  if (!ctx.history) return;
  if (ctx.current_round % print_every_ != 0) return;

  // Print all metrics for this round
  for (const auto& r : ctx.history->results) {
    if (r.round == ctx.current_round) {
      Logger::instance().log(
        LogLevel::INFO,
        "Round metric",
        {
          {"event", "round_metric"},
          {"round", std::to_string(r.round)},
          {"dataset", r.dataset_name},
          {"metric", r.metric_name},
          {"value", std::to_string(r.value)}
        });
    }
  }
}

} // namespace xgb
