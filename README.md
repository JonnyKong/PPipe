# PPipe

[ATC '25] Pipe: Efficient Video Analytics Serving on Heterogeneous GPU Clusters via Pool-Based Pipeline Parallelism

### Dependencies and Build

* Install dependencies

```
pip install -r requirements.txt
```

* Build Java simulator

```bash
cd cluster-sim
./gradlew installDist
```

### MILP solver

* **Prepartition MILP (sec 5.2)**

```bash
python milp_solver/prepartition_ilp.py
```

This writes the outputs to `outputs/prepartition_mappings`, one CSV for each
model. Each CSV represents the mapping between DNN layer names to its
corresponding chunk ID.

For reference, we've uploaded the expected outputs to
`data/prepartition_mappings`.

* **Main MILP** (ETA: 5mins)

```bash
python milp_solver/run_ilp_v4_in_batch.py maf19
python milp_solver/run_ilp_v4_in_batch.py maf21
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
