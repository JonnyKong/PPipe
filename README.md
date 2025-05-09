# PPipe

[ATC '25] Pipe: Efficient Video Analytics Serving on Heterogeneous GPU Clusters via Pool-Based Pipeline Parallelism

### Codebase structure

```
.
├── cluster-sim             # The discrete-event simulator (sec 6)
│
├── data
│   ├── maf_traces              # Microsoft MAF request traces
│   ├── model_list.txt          # List of DNNs to be used
│   ├── models                  # Model latencies and intermediate sizes from offline profiling
│   ├── plans                   # MILP plans
│   └── prepartition_mappings   # Pre-partition MILP (sec 5.2) output
│
├── milp_solver             # The MILP solver
│   ├── contract_layers.py
│   ├── group_dnns.py
│   ├── prepartition_ilp.py     # Prepartition MILP implementation (sec 5.2)
│   ├── ilp_v4.py               # Main MILP implementation (Appendix A)
│   └── run_ilp_v4_in_batch.py  # Driver script to run the main MILP
│
├── outputs                 # Artifact outputs will be written here
│   └── README.md
│
└── scripts
    ├── parse_cluster_sim.py
    ├── plot_utils.py
    ├── plot.py
    ├── run_sim_in_batch.py
    ├── run_simulator.sh
    └── sim_config.py

```

### Installation

* Download code and data

```bash
git clone https://github.com/JonnyKong/PPipe
cd PPipe
git lfs install
git lfs pull
```

* Install dependencies

```bash
conda create -n ppipe python=3.12   # Please replace env name with a unique one
conda activate ppipe
pip install -r requirements.txt
```

* Build Java simulator

```bash
cd cluster-sim
./gradlew installDist
cd ..
```

### MILP solver

* **Prepartition MILP (sec 5.2)** (ETA: 5 mins)

```bash
python milp_solver/prepartition_ilp.py
```

This writes the outputs to `outputs/prepartition_mappings`, one CSV for each
model. Each CSV represents the mapping between DNN layer names to its
corresponding chunk ID.

For reference, we've uploaded the expected outputs to
`data/prepartition_mappings`.

* **Main MILP** (ETA: 5 mins)

```bash
python milp_solver/run_ilp_v4_in_batch.py main_maf19
python milp_solver/run_ilp_v4_in_batch.py main_maf21
```

This computes an MILP plan for each group of 3 DNNs, for NP, DART-r, and PPipe
respectively. The outputs are written to `outputs/plans/maf[19|21]`, one json
file for each DNN group.

For reference, we've uploaded the expected outputs to `data/plans`.

* **Ablation MILP**
```bash
python milp_solver/run_ilp_v4_in_batch.py ablation
```

### Figures 6, 7, 8

```bash
# Main results on MAF 19 traces (ETA: 10 mins)
./scripts/run_simulator.sh main_results_maf19
# Main results on MAF 21 traces (ETA: 10 mins)
./scripts/run_simulator.sh main_results_maf21
# Plot the figures (ETA: <1 min)
python scripts/plot.py fig6
python scripts/plot.py fig7
python scripts/plot.py fig8
```

### Fig 10

```bash
# Ablation study
./scripts/run_simulator.sh ablation_results_maf19
# Plot the figures (ETA: <1 min)
python scripts/plot.py fig10
```
