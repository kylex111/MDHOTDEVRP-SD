# MDHOTDEVRP-SD

This repository contains the GBGA-MOVNS algorithm and experiment configurations for the MDHOTDEVRP-SD. The algorithm searches for routing solutions that balance operating cost and accumulated range anxiety (RA) under stochastic demand.

## Project files

```text
run.py                         Command-line entry point
configs/                       Experiment settings in JSON
environment/                   Saved experiment environments and dependency versions
seeds/                         Recorded seeds and their source locations
src/
  configuration/               Experiment parameters and random seed generation
  domain/                      Problem instances, nodes, and route data
  evaluation/                  Demand scenarios, route decoding, energy, cost, and RA
  algorithms/
    spr.py                     GBGA-MOVNS and its ablation variants
    grouping/                  Customer clusters and granular balls
    initialization/            Genetic algorithms for initial route construction
    search/                    Neighborhoods, VND, and Pareto archives
    moalns/                    Adapted multi-objective ALNS baseline
    spea/                      SPEA2 components
    pymoo_solver.py            pymoo comparison algorithms
  experiments/                 Single runs, batches, tuning, and result storage
  analysis/                    Statistics, quality metrics, and plots
```

The main algorithm is implemented in [spr.py](src/algorithms/spr.py). The [evaluator](src/evaluation/evaluator.py) calculates route objectives across demand scenarios. The [route decoder](src/evaluation/causal.py) simulates vehicle movements, reveals demand on arrival, and arranges charging or replenishment when required.

## Benchmark instances

Download the benchmark instances from the [instance repository](https://github.com/kylex111/MDHOTDEVRP-SD-instances). Create an `instances/` directory in the root of this project and place the downloaded `.txt` files directly inside it. For example, the C103 file should be at `instances/c103_21.txt`.

## Run an experiment

Run the following commands from the repository root. The first command lists the available options. The second runs GBGA-MOVNS once on C103 using the default configuration and saves the results to `results/c103_run001`.

```powershell
python run.py --help
python run.py single --config configs/defaults/experiment.json --instance instances/c103_21.txt --run-id 1 --out results/c103_run001
```
