#include "sle/full_engine.hpp"
#include <iostream>
int main(){
  std::vector<sle::TrainingExample> dataset;
  sle::BitVector in(1); in.set(0,false);
  sle::BitVector out(1); out.set(0,false);
  dataset.push_back({{in}, out});
  sle::FullEngineConfig cfg;
  cfg.gate_count = 3;
  cfg.solver_policy.tiers = {
    {sle::SynthesisTier::Exact, false, 0, true},
    {sle::SynthesisTier::Local, false, 0, true},
    {sle::SynthesisTier::MonteCarloTreeSearch, false, 0, true},
  };
  auto r = sle::train_full_engine(dataset, cfg, {});
  std::cout << r.fitness.total << "\n";
}
