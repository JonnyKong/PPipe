import bisect
import copy
import itertools
import json
import time
from dataclasses import asdict
from pathlib import Path

import gurobipy as gp
import numpy as np
import pandas as pd
from group_dnns import get_random_grouping
from gurobipy import GRB
from ilp_v4 import get_datadir
from ilp_v4 import MultitaskCluster
from ilp_v4 import MultitaskConfig
from ilp_v4 import read_runtime_from_blockwise_profiling
from ilp_v4 import read_runtime_from_layerwise_profiling
from ilp_v4 import RuntimeFmt


def get_dnn_runtime(dnn):
    with open(
        get_datadir()
        / f"models/model-profile-tf32/{dnn}/L4-1/1/{dnn}-L4.engine.timing.json"
    ) as f:
        runtimes_ms = json.load(f)
    runtime_mean_us = np.mean([r["computeMs"] for r in runtimes_ms]) * 1000
    return runtime_mean_us


def get_baseline_plan(cfg: MultitaskConfig, timelimit=None):
    """
    Return the baseline xput and json plan.
    """
    xput_total = 0.0

    plan = {
        "xput": None,
        "config": asdict(cfg.get_singletask_cfgs()[0]),
        "pipelines": [],
        "sla": cfg.sla_arr[0],
    }

    for gpu_name, gpu_limit in zip(cfg.gpu_name_arr, cfg.gpu_limit_arr):
        if cfg.runtime_fmt == RuntimeFmt.BLOCKWISE:
            lat_table = read_runtime_from_blockwise_profiling(
                cfg.dnn_name_arr[0], gpu_name, 0
            )
        elif cfg.runtime_fmt == RuntimeFmt.LAYERWISE_W_PREPARTITION:
            lat_table = read_runtime_from_layerwise_profiling(cfg.dnn_name_arr[0], gpu_name, 0)
        else:
            raise NotImplementedError()
        lat_infer_arr = np.sum(lat_table.to_numpy(), axis=0)

        # Add batch building time to runtime
        if cfg.est_xput_arr:
            est_batch_build_lat_arr = (
                cfg.batch_build_factor
                * 1e6
                * np.arange(len(lat_infer_arr))
                / cfg.est_xput_arr[0]
            )
            est_batch_build_lat_arr = np.rint(est_batch_build_lat_arr).astype(int)
        else:
            est_batch_build_lat_arr = [0 for _ in range(len(lat_infer_arr))]
        lat_arr = lat_infer_arr + est_batch_build_lat_arr
        lat_arr += cfg.hist_adjustment_arr[0] + cfg.hist_adjustment_w_scheduling_arr[0]

        # Use the sum of inference and batch building latency to calculate the largest possible bs
        bs = bisect.bisect_left(lat_arr, cfg.sla_arr[0])
        if bs == 0:
            continue

        # But still uses only inference time to calculate xput, because GPU inference and batch
        # building can pipeline
        lat_infer = lat_infer_arr[bs - 1]
        xput = gpu_limit * bs / (lat_infer / 1e6)
        xput_total += xput
        plan["pipelines"].append(
            {
                "xput": xput,
                "partitions": [
                    {
                        "dnn": cfg.dnn_name_arr[0],
                        # TODO: remove hardcoded layers
                        "layers": [
                            0,
                            len(cfg.get_singletask_cfgs()[0].transmit_time_us_arr) + 1,
                        ],
                        "gpu": gpu_name,
                        "mps": 0,
                        "bs": bs,
                        "num_gpu": gpu_limit,
                        "lat_infer": int(lat_infer),
                        "lat_trans": 0.0,
                    }
                ],
                "est_xput": cfg.est_xput_arr[0] if cfg.est_xput_arr else None,
                "est_batch_build_lat": int(np.rint(est_batch_build_lat_arr[bs - 1])),
                "batch_build_factor": cfg.batch_build_factor,
                "hist_adjustment": cfg.hist_adjustment_arr[0],
                "hist_adjustment_w_scheduling": cfg.hist_adjustment_w_scheduling_arr[0],
            }
        )

    plan["xput"] = xput_total
    if xput_total > 0:
        return [plan]
    else:
        return []


def get_baseline_multitask_plan(cfg: MultitaskConfig, timelimit=1800, logfile=None):
    cfg = copy.deepcopy(cfg)
    cfg.max_num_parts = 1
    cfg.num_mps_levels = [1 for _ in cfg.gpu_name_arr]
    return get_ilp_plan(cfg)


def get_ilp_plan(cfg: MultitaskConfig, timelimit=1800, logfile=None, mipgap=None):
    model = gp.Model("cluster_serving")
    c = MultitaskCluster(model, cfg)

    if timelimit:
        model.Params.TimeLimit = timelimit
    if logfile:
        model.Params.LogFile = logfile
    if mipgap:
        model.Params.MIPGap = mipgap

    model.setObjective(c.xput, GRB.MAXIMIZE)
    model.Params.AggFill = 10

    start = time.time()
    model.optimize()
    end = time.time()
    runtime = end - start
    print(f"Runtime: {runtime}")

    try:
        plan = c.serialize()
        plan[0]["mipgap"] = model.MIPGap
        plan[0]["runtime"] = runtime
    except AttributeError:
        # Model stuck in presolve stage w/o solution might also return GRB.TIME_LIMIT
        plan = {}
    return plan


def get_even_prepart_plan(cfg: MultitaskConfig, timelimit=1800, logfile=None):
    cfg = copy.deepcopy(cfg)
    cfg.runtime_fmt = RuntimeFmt.LAYERWISE_W_EVEN_RUNTIME_PREPARTITION
    return get_ilp_plan(cfg, timelimit, logfile)


def get_dart_plan(cfg: MultitaskConfig, timelimit):
    configs_and_counts = gen_dart_configs_and_counts_from_full_config(cfg)
    plans_and_counts = [
        (get_ilp_plan(cfg_, timelimit), count) for cfg_, count in configs_and_counts
    ]
    plan = merge_plans(plans_and_counts)
    plan[0]["config"] = asdict(cfg.get_singletask_cfgs()[0])
    return plan


def gen_dart_configs_and_counts_from_full_config(cfg: MultitaskConfig):
    gpu_id_excessive = np.argmax(cfg.gpu_limit_arr)
    num_1x1_cfg = np.min(cfg.gpu_limit_arr)
    num_1_cfg = np.max(cfg.gpu_limit_arr) - num_1x1_cfg

    # 1x1 plan
    cfg_1x1 = copy.deepcopy(cfg)
    cfg_1x1.gpu_limit_arr = [1, 1]
    cfg_1x1.num_mps_levels = [1, 1]

    # 1 plan
    cfg_1 = copy.deepcopy(cfg)
    cfg_1.gpu_name_arr = [cfg_1.gpu_name_arr[gpu_id_excessive]]
    cfg_1.num_mps_levels = [1]
    cfg_1.gpu_limit_arr = [1]
    cfg_1.max_num_parts = 1
    cfg_1.num_gpu_per_server_arr = [cfg_1.num_gpu_per_server_arr[gpu_id_excessive]]

    return [
        (cfg_1x1, num_1x1_cfg),
        (cfg_1, num_1_cfg),
    ]


def merge_plans(plans_and_counts):
    plan_merged = copy.deepcopy(plans_and_counts[0][0])
    plan_merged[0]["xput"] = 0
    plan_merged[0]["pipelines"] = []
    plan_merged[0]["config"] = {}  # Do not know how to merge here, leave to caller

    for plan, count in plans_and_counts:
        if count > 0:
            plan = horizontal_scale_pipelines(plan, count)
            plan_merged[0]["xput"] += plan[0]["xput"]
            plan_merged[0]["pipelines"].extend(plan[0]["pipelines"])
    return plan_merged


def horizontal_scale_pipelines(plan, count):
    plan_duplicated = copy.deepcopy(plan)
    plan_duplicated[0]["xput"] *= count

    for pipeline in plan_duplicated[0]["pipelines"]:
        pipeline["xput"] *= count
        for part in pipeline["partitions"]:
            part["xput"] *= count
            part["num_gpu"] *= count
    return plan_duplicated


def get_multidnn_dart_plan(cfg: MultitaskConfig, timelimit=1800):
    """
    First run baseline ILP to divide the GPUs among DNNs. Then run DART-r for
    each DNN separately. GPUs not allocated in the baseline gets assigned
    uniformly to DNNs.
    """
    baseline_plan = get_baseline_multitask_plan(cfg)

    # Count the number of GPUs used by each DNN in the baseline plan
    gpu_used_arr_arr = [[0 for gpu in cfg.gpu_name_arr] for dnn in cfg.dnn_name_arr]

    for dnn_id, dnn_plan in enumerate(baseline_plan):
        # Double check dnns are still in order
        assert dnn_plan["config"]["dnn_name"] == cfg.dnn_name_arr[dnn_id]

        for pipeline in dnn_plan["pipelines"]:
            assert len(pipeline["partitions"]) == 1
            gpu = pipeline["partitions"][0]["gpu"]
            num_gpu = int(round(pipeline["partitions"][0]["num_gpu"]))
            gpu_id = cfg.gpu_name_arr.index(gpu)
            gpu_used_arr_arr[dnn_id][gpu_id] += num_gpu

    # Count how many gpus are remaining
    gpu_used_arr = np.sum(np.array(gpu_used_arr_arr, dtype=int), axis=0)
    gpu_remaining_arr = np.array(cfg.gpu_limit_arr, dtype=int) - gpu_used_arr
    print("gpu_remaining_arr: ", gpu_remaining_arr)

    # Distribute remaining GPUs uniformly among DNNs
    gpu_remaining_each_dnn = gpu_remaining_arr // len(cfg.dnn_name_arr)
    gpu_limit_arr_arr = np.array(gpu_used_arr_arr, dtype=int) + gpu_remaining_each_dnn
    print("gpu_limit_arr_arr: ", gpu_limit_arr_arr)

    # Create a new cfg one for each DNN
    plans = []
    for dnn_id in range(len(cfg.dnn_name_arr)):
        sub_cfg = copy.deepcopy(cfg)
        sub_cfg.dnn_name_arr = [cfg.dnn_name_arr[dnn_id]]
        sub_cfg.gpu_limit_arr = gpu_limit_arr_arr[dnn_id].tolist()
        sub_cfg.sla_arr = [cfg.sla_arr[dnn_id]]
        sub_cfg.workload_weights = [cfg.workload_weights[dnn_id]]
        if cfg.est_xput_arr:
            sub_cfg.est_xput_arr = [cfg.est_xput_arr[dnn_id]]
        sub_cfg.hist_adjustment_arr = [cfg.hist_adjustment_arr[dnn_id]]
        sub_cfg.hist_adjustment_w_scheduling_arr = [
            cfg.hist_adjustment_w_scheduling_arr[dnn_id]
        ]
        plan = get_dart_plan(sub_cfg, timelimit=timelimit)
        plans.append(plan[0])
    return plans


def plan_is_empty(plan):
    return len(plan) == 0


def main_multitask_v2(sla_discount=0.0, group_size=3, workload_weights=None,
                      savedir=Path('./plans')):
    """
    For NSDI '25, group DNNs into groups of 3.

    For the ATC' 25 submission, we additionally specify a `workload_weights`,
    which overrides NSDI's equal workload weights. The workload_weights is
    offline calculated in `process_maf_functions_trace.py`, by assigning the
    first hour of the MAF '19 trace to 3 DNNs round-robin, which gave a weight
    ratio of [0.30, 0.33, 0.37].
    """
    dnn_name_arr = pd.read_csv("./data/model_list.txt", sep=" ", header=None)[0].to_numpy()

    def get_2gpu_configs():
        dnn_name_arr_arr = get_random_grouping(dnn_name_arr, group_size=group_size, seed=17)
        clusters = [
            [["V100", "T4"], [25, 75], True, 2, [4, 4], [4, 2], 10],
            [["V100", "P4"], [25, 75], True, 2, [4, 1], [2, 1], 6],
            [["L4", "T4"], [25, 75], True, 2, [4, 4], [4, 2], 10],
            [["L4", "P4"], [25, 75], True, 2, [4, 1], [1, 1], 6],
        ]
        return list(itertools.product(dnn_name_arr_arr, clusters))

    configs = get_2gpu_configs()    # Uncomment for 2-gpu setup
    sla_multiplier = 5
    if workload_weights is None:
        workload_weights = [1.0 for _ in range(group_size)]

    df = pd.DataFrame()

    for dnn_combinations, cluster in configs:
        (
            gpu_name_arr,
            gpu_limit_arr,
            force_sum_gpu_integer_per_partition,
            max_parts,
            num_mps_levels,
            num_gpu_per_server_arr,
            bw_gbps,
        ) = cluster
        sla_arr = [get_dnn_runtime(dnn) * sla_multiplier for dnn in dnn_combinations]
        hist_adjustment_arr = [round(sla_discount * sla) for sla in sla_arr]
        hist_adjustment_w_scheduling_arr = [0 for _ in dnn_combinations]

        tag = "-".join(dnn_combinations)
        tag += f'_{"-".join([str(w) for w in workload_weights])}'
        tag += f'_{"-".join(gpu_name_arr)}'
        tag += f'_{"-".join([str(s) for s in gpu_limit_arr])}'
        tag += f"_bw-gbps-{bw_gbps}"
        tag += f"_sla-multiplier-{sla_multiplier}"

        # Estimate an approximate xput
        cfg = MultitaskConfig(
            list(dnn_combinations),
            gpu_limit_arr,
            gpu_name_arr,
            sla_arr,
            workload_weights,
            num_mps_levels,
            max_parts,
            True,
            None,
            1.3,
            hist_adjustment_arr,
            hist_adjustment_w_scheduling_arr,
            num_gpu_per_server_arr,
            force_sum_gpu_integer_per_partition,
            bw_gbps,
            RuntimeFmt.BLOCKWISE,
        )
        plan_bl_xput_est = get_baseline_multitask_plan(cfg)
        est_xput_arr = [p["xput"] for p in plan_bl_xput_est]
        cfg.est_xput_arr = est_xput_arr

        row = {
            "dnn_combinations": list(dnn_combinations),
            "sla_arr": sla_arr,
            "gpu_name_arr": str(gpu_name_arr),
            "gpu_limit_arr": str(gpu_limit_arr),
            "workload_weights": str(workload_weights),
            "num_gpu_per_server_arr": str(num_gpu_per_server_arr),
            "force_sum_gpu_integer_per_partition": force_sum_gpu_integer_per_partition,
            "bw_gbps": bw_gbps,
        }
        scheduler_fns = [
            ("bl", get_baseline_multitask_plan),
            ("v4", get_ilp_plan),
        ]
        if len(gpu_name_arr) == 2:
            scheduler_fns.append(("dart-r", get_multidnn_dart_plan))

        for scheduler, fn in scheduler_fns:
            plan = fn(cfg, timelimit=60)
            if not plan_is_empty(plan):
                row |= {
                    f"xput_{scheduler}_arr": [p["xput"] for p in plan],
                    f"xput_{scheduler}_norm": get_norm_xput(
                        [p["xput"] for p in plan], workload_weights
                    ),
                    f"plan_{scheduler}": json.dumps(plan, indent=2),
                }
            savedir.mkdir(exist_ok=True, parents=True)
            with open(savedir / f"{tag}_{scheduler}.json", "w") as f:
                json.dump(plan, f, indent=2)

        df = pd.concat([df, pd.json_normalize(row)])
        df.to_csv("outputs/plans/milp_summary.csv", index=False)


def get_norm_xput(xput_arr, workload_weights):
    return np.min([x / w for x, w in zip(xput_arr, workload_weights)])


if __name__ == "__main__":
    # MAF '19
    main_multitask_v2(sla_discount=0.4, group_size=3,
                      workload_weights=[0.30, 0.33, 0.37],
                      savedir=Path('outputs/plans/maf19'))
    # MAF '21
    main_multitask_v2(sla_discount=0.4, group_size=3,
                      workload_weights=[0.39, 0.26, 0.35],
                      savedir=Path('outputs/plans/maf21'))
