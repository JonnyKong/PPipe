# PPipe

[USENIX ATC '25] PPipe: Efficient Video Analytics Serving on Heterogeneous GPU
Clusters via Pool-Based Pipeline Parallelism

### Codebase structure

```
.
├── cluster-sim             # The discrete-event simulator (Sec. 6)
│
├── data
│   ├── maf_traces              # Microsoft MAF request traces
│   ├── model_list.txt          # List of DNN models
│   ├── models                  # Offline profiling results
│   │   ├── cuts-no-const           # Layer & edge names
│   │   ├── model-profile-tf32      # Profiled latency per layer in TensorRT
│   │   ├── node-profile-no-const   # Profiled latency per layer in ONNX
│   │   ├── block-timing-tf32       # Profiled latency per pre-partitioned block in TensorRT
│   │   └── shapes                  # Shapes and sizes of intermediate representations
│   │   
│   ├── plans                   # Reference partition plans from the MILP
│   └── prepartition_mappings   # Reference prepartition MILP outputs (Sec. 5.2)
│
├── milp_solver             # The MILP solver
│   ├── contract_layers.py
│   ├── group_dnns.py
│   ├── prepartition_ilp.py     # Prepartition MILP implementation (Sec. 5.2)
│   ├── ilp_v4.py               # Main MILP formulation (Appendix A)
│   └── run_ilp_v4_in_batch.py  # Batch driver script for main MILP
│
├── outputs                 # Outputs from running the artifact
│   └── README.md
│
└── scripts                 # Scripts for automating experiments, plotting figures, etc.
    ├── parse_cluster_sim.py
    ├── plot_utils.py
    ├── plot.py
    ├── run_sim_in_batch.py
    ├── run_simulator.sh
    └── sim_config.py

```

### Installation

1. Clone the repository and pull LFS-tracked files:

```bash
git clone https://github.com/JonnyKong/PPipe
cd PPipe
git lfs install
git lfs pull
```

2. Install dependencies:

```bash
conda create -n ppipe python=3.12   # Please choose a different environment name
conda activate ppipe
pip install -r requirements.txt
```

3. Set up a [Gurobi
   license](https://www.gurobi.com/academia/academic-program-and-licenses/)

  * [**Note for ATC'25 artifact evaluation reviewers**]: A preconfigured
    machine with credentials will be made available via HotCRP.

4. Build the Java-based simulator:

```bash
cd cluster-sim
./gradlew installDist
cd ..
```

### Reproducing the paper

#### 1. Running the MILP Solver to Generate Partition Plans

**1.1 Prepartition MILP (Sec. 5.2)** -- ETA: 10 mins

```bash
python milp_solver/prepartition_ilp.py
```

* Outputs are written to `outputs/prepartition_mappings/`, one CSV per model.
* Each CSV maps DNN layer names to their corresponding chunk IDs.
* Reference outputs are available in `data/prepartition_mappings/`.

**1.2 MILP for the Main Results** -- ETA: 15 mins

```bash
python milp_solver/run_ilp_v4_in_batch.py main_maf19
python milp_solver/run_ilp_v4_in_batch.py main_maf21
```
* Computes MILP plans for groups of 3 DNNs using NP, DART-r, and PPipe.
* Outputs are saved under `outputs/plans/maf19/` and `outputs/plans/maf21/`. 
* Reference outputs are included in `data/plans/`.
* Each plan describes how a group 3 DNNs are deployed on a cluster of 100 GPUs
  in JSON format, with the following notable fields:
  ```
  [
    {                                                       // <- DNN 0
      "xput": <throughput for DNN 0>
      "pipelines": [                                        // <- DNN 0 pipeline 0
        {
          "xput": <xput>,           // throughput of this pipeline
          "partitions": [                                   // <- DNN 0 pipeline 0 partition 0
            "layers": [             // which prepartition chunks are in this partition
              start_chunk_id,
              end_chunk_id
            ],
            "gpu": <gpu>,           // e.g., V100, L4,
            "mps": <mps>,           // e.g., 1 (100%), 2 (50%), etc.
            "num_gpu": <num_gpu>,   // number of GPUs assigned, can be fractional if mps > 1
            "lat_infer": <lat_infer>,   // partition inference latency
            "lat_trans": <lat_trans>,   // transmission latency to next partition
          ]
        },
        ...
      ]
    },
    ...
  ]
  ```

**1.3 MILP for the Ablation Study** (ETA: 10 mins)

```bash
python milp_solver/run_ilp_v4_in_batch.py ablation
```

* Outputs are saved under `outputs/plans/ablation`.

#### 2. Running the Discrete-event Simulator using the MILP-generated Plans

```bash
# Main results on MAF 19 traces (ETA: 20 mins)
./scripts/run_simulator.sh main_results_maf19
# Main results on MAF 21 traces (ETA: 20 mins)
./scripts/run_simulator.sh main_results_maf21
# Ablation study (ETA: 20 mins)
./scripts/run_simulator.sh ablation_results_maf19
```

* For each MILP plan, we iterates over decreasing load factors from 1.0 (step
  size 0.5), run a inference session using each load factor, until 99% SLO
  attainment is achieved.
* The outputs will be written to `outputs/cluster-logs/`, with folder names
  formatted as:
    * `<dnns>_<gpus>_<gpu-counts>_<bandwidth>_sla-multiplier-5_<algo>`, where:
        * `algo=bl` -> NP
        * `algo=dart-r` -> DART-r
        * `algo=v4` -> PPipe
* Each session produces multiple CSV files:
    * `master.csv`: frontend load balancer output (each row is a batch)
    * `i-j.csv`: output of partition j on pipeline i, forwarded to the next
      partition if exists (each row is a batch)
* For each set of experiments, a CSV will be produced with each row
  representing a session:
    * `outputs/cluster-logs/maf19/logs.csv`
    * `outputs/cluster-logs/maf21/logs.csv`
    * `outputs/cluster-logs/ablation/logs.csv`

#### 3. Reproducing the Figures

```bash
# Plot the figures (ETA: <1 min)
python scripts/plot.py fig6
python scripts/plot.py fig7
python scripts/plot.py fig8
python scripts/plot.py fig10
```

* The figures will be written to `outputs/`
