# PPipe

[ATC '25] Pipe: Efficient Video Analytics Serving on Heterogeneous GPU Clusters via Pool-Based Pipeline Parallelism

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

* **Main MILP**

```bash
python milp_solver/run_ilp_v4_in_batch.py
```

This writes the outputs to `outputs/plans`, one json file for each cluster.

For reference, we've uploaded the expected outputs to
`data/plans`.

### Simulator setup
```bash
cd cluster-sim
./gradlew installDist
```

### Figure 6
