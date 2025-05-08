import json
import random
import sys
from typing import List

import pandas as pd


def get_random_grouping(dnn_name_arr, group_size, seed=0):
    assert len(dnn_name_arr) % group_size == 0

    dnn_name_arr = sorted(dnn_name_arr)
    random.seed(seed)
    random.shuffle(dnn_name_arr)

    grouping = []
    for i in range(0, len(dnn_name_arr), group_size):
        grouping.append(tuple(sorted(dnn_name_arr[i: i + group_size])))
    return grouping


def get_plan_name(dnn_str, gpu_models, gpu_counts, bw, scheduler="v4"):
    cluster_str = "-".join(gpu_models) + "_" + "-".join(str(c) for c in gpu_counts)
    return f"{dnn_str}_{cluster_str}_bw-gbps-{bw}_sla-multiplier-5_{scheduler}"


def multitask_configs(plan_root, log_root,
                      workload_weights: List[str],
                      use_trace_arrival):
    dnn_name_arr = pd.read_csv(
        "data/model_list.txt", sep=" ", header=None
    )[0].to_numpy()
    dnn_name_arr_arr = get_random_grouping(dnn_name_arr, group_size=3, seed=17)

    group_size = len(dnn_name_arr_arr[0])
    assert len(workload_weights) == group_size

    clusters = [
        [["L4", "T4"], [25, 75], 10],
        [["L4", "P4"], [25, 75], 6],
        [["V100", "T4"], [25, 75], 10],
        [["V100", "P4"], [25, 75], 6],
    ]

    for dnn_group_id, dnn_group in enumerate(dnn_name_arr_arr):
        for gpu_models, gpu_counts, bw in clusters:
            for dnn_id, dnn in enumerate(dnn_group):
                for scheduler in ["v4", "bl", "dart-r"]:
                    dnn_str = (
                        "-".join(sorted(dnn_group))
                        + "_"
                        + "-".join(workload_weights)
                    )
                    plan_name = get_plan_name(
                        dnn_str, gpu_models, gpu_counts, bw, scheduler
                    )
                    plan_path = plan_root / f"{plan_name}.json"

                    # Empty plan means ILP no solution
                    if not plan_path.exists():
                        print(f"Missing plan: {plan_path}", file=sys.stderr)
                        continue
                    if scheduler == "v4":
                        with open(plan_path, encoding="utf-8") as f:
                            xput = int(json.load(f)[dnn_id]["xput"])

                    # Make sure the ordering of the DNNs in the plan matches the
                    # order in the filename
                    with open(plan_path) as f:
                        plan = json.load(f)
                    assert tuple([p["config"]["dnn_name"] for p in plan]) == dnn_group

                    log_dir = log_root / plan_name / f"dnn-id-{dnn_id}"
                    yield [
                        plan_path,
                        dnn,
                        log_dir,
                        xput,
                        dnn_id,
                        "RESERVATION",
                        use_trace_arrival,
                        dnn_group_id,
                        gpu_models,
                        gpu_counts,
                        bw,
                        scheduler,
                    ]


def ablation_configs(plan_root, log_root, use_trace_arrival):
    dnn_name_arr = pd.read_csv(
        "scripts/trt/model-zoo/selected_models.txt", sep=" ", header=None
    )[0].to_numpy()

    clusters = [
        [["L4", "T4"], [25, 75], 10],
    ]
    for i, dnn in enumerate(dnn_name_arr):
        for gpu_models, gpu_counts, bw in clusters:
            for scheduler in ["v4", "bl", "nexus"]:
                plan_name = get_plan_name(
                    dnn,
                    gpu_models,
                    gpu_counts,
                    bw,
                    ("bl" if scheduler == "bl" else "v4"),
                )
                plan_path = plan_root / f"{plan_name}.json"

                # Empty plan means ILP no solution
                if not plan_path.exists():
                    print(f"Missing plan: {plan_path}", file=sys.stderr)
                    continue
                if scheduler == "v4":
                    with open(plan_path, encoding="utf-8") as f:
                        xput = int(json.load(f)[0]["xput"])

                dnn_id = 0
                dnn_group_id = i
                # Re-compute plan name without converting "nexus" to "v4"
                plan_name = get_plan_name(
                    dnn,
                    gpu_models,
                    gpu_counts,
                    bw,
                    scheduler,
                )
                log_dir = log_root / plan_name / f"dnn-id-{dnn_id}"
                yield [
                    plan_path,
                    dnn,
                    log_dir,
                    xput,
                    dnn_id,
                    "SLA_AWARE" if scheduler == "nexus" else "RESERVATION",
                    use_trace_arrival,
                    dnn_group_id,
                    gpu_models,
                    gpu_counts,
                    bw,
                    scheduler,
                ]
