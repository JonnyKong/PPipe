import argparse
import json
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

import sim_config


def parse_cluster_sim_ilppartitioned(log_dir, sla, plan, duration=30000000):
    df_master = pd.read_csv(Path(log_dir) / "master.csv")

    # Find out how many streams there are, to calculate the duration of the rampup region
    num_streams_planned = plan["xput"] / 30
    num_streams = len(df_master.streamId.value_counts())
    startup_us = num_streams * 1000  # 1000 is mean of client inter-arrival time
    df_master = df_master[
        (df_master.arrivalTime >= startup_us) & (df_master.arrivalTime < duration)
    ]

    # Read logs of between-partition LBs into a 2d array
    df_lb_arr_arr = []
    num_pipelines = len(plan["pipelines"])
    for pipeline_id in range(num_pipelines):
        df_lb_arr = [
            pd.read_csv(p)
            for p in sorted(Path(log_dir).glob(f"{pipeline_id}-[0-9]*.csv"))
        ]
        df_lb_arr = [
            df[(df.arrivalTime >= startup_us) & (df.arrivalTime < duration)]
            for df in df_lb_arr
        ]
        df_lb_arr_arr.append(df_lb_arr)

    # Only the master LB drops requests
    if len(df_master) > 0:
        perc_dropped = (
            df_master.discarded.sum()
            + sum(df.discarded.sum() for dfs in df_lb_arr_arr for df in dfs)
        ) / len(df_master)
    else:
        perc_dropped = 1.0

    # Only the master LB has batch building time
    lat_batch_build_arr = df_master.timeArriveWorker - df_master.timeArriveLb
    lat_batch_build_mean = lat_batch_build_arr.mean()
    lat_batch_build_p99 = lat_batch_build_arr.quantile(0.99)

    # Max queue len at master lb
    max_queue_len = df_master.queueLen.max()

    # Perc. exceed sla (out of un-dropped requests)
    num_violated_sla = 0
    for pipeline_id in range(num_pipelines):
        if len(df_lb_arr_arr[pipeline_id]) == 0:
            # For partitions of length 1, look at master LB to compute SLA violation
            df = df_master
        else:
            # Otherwise, look at the LB before the last partition to compute SLA violation
            df = df_lb_arr_arr[pipeline_id][-1]
        num_violated_sla += len(
            df[(df.pipelineId == pipeline_id) & (df.timeComplete > df.deadline)]
        )
    if len(df_master):
        perc_violate_sla = num_violated_sla / len(df_master)
    else:
        perc_violate_sla = 1.0

    # Return queuing time as a string
    lat_queue_master = df_master.timeStartInference - df_master.timeArriveWorker
    lat_queue_mean_str = f"{lat_queue_master.mean():.1f}"
    lat_queue_p99_str = str(lat_queue_master.quantile(0.99))
    for pipeline_id in range(num_pipelines):
        lat_queue_mean_str += "\n- " + str(
            [
                (df.timeStartInference - df.timeArriveWorker).mean()
                for df in df_lb_arr_arr[pipeline_id]
            ]
        )
        lat_queue_p99_str += "\n- " + str(
            [
                (df.timeStartInference - df.timeArriveWorker).quantile(0.99)
                for df in df_lb_arr_arr[pipeline_id]
            ]
        )

    # Calulcate queuing P99
    df = df_master[["requestId"]].copy()
    df["queuing"] = df_master.timeStartInference - df_master.timeArriveWorker
    df = df.set_index("requestId").sort_index()
    for pipeline_id in range(num_pipelines):
        for df_ in df_lb_arr_arr[pipeline_id]:
            df_["queuing"] = df_.timeStartInference - df_.timeArriveWorker
            df_ = df_[["requestId", "queuing"]].set_index("requestId").sort_index()
            df = df.add(df_, fill_value=0.0)
    lat_queue_total_mean = df.queuing.mean()
    lat_queue_total_p99 = df.queuing.quantile(0.99)

    # Slack in plan, averaged over partitions weighted by xput
    elapsed_slack = 0
    elapsed_xput = 0
    elapsed_est_batch_build_lat = 0
    for pipeline in plan["pipelines"]:
        slack = sla
        for partition in pipeline["partitions"]:
            slack -= partition["lat_infer"]
            slack -= partition["lat_trans"]
        elapsed_est_batch_build_lat += (
            pipeline["est_batch_build_lat"] * pipeline["xput"]
        )
        elapsed_slack += slack * pipeline["xput"]
        elapsed_xput += pipeline["xput"]
    if elapsed_xput:
        plan_slack = elapsed_slack / elapsed_xput
        est_batch_build_lat = elapsed_est_batch_build_lat / elapsed_xput
    else:
        plan_slack = 0
        est_batch_build_lat = 0

    # Planned and actual BS for each pipeline
    bs_planned_arr = []
    bs_actual_arr = []
    for pipeline_id in range(num_pipelines):
        # Assuming bs same across partitions for now
        bs_planned_arr.append(plan["pipelines"][pipeline_id]["partitions"][0]["bs"])
        df_this_pipeline = df_master[
            (df_master.pipelineId == pipeline_id) & (~df_master.discarded)
        ]
        bs_actual_arr.append(np.round(df_this_pipeline.bs.mean(), 2))

    # Percentage of each drop cause
    num_dropped = df_master.discarded.sum()
    if num_dropped > 0:
        drop_cause_perc_arr = []
        for drop_cause in range(3):
            num_dropped_ = len(
                df_master[df_master.discarded & (df_master.dropCause == drop_cause)]
            )
            drop_cause_perc_arr.append(num_dropped_ / num_dropped)
    else:
        drop_cause_perc_arr = [0.0 for _ in range(3)]

    # Return the GPU utilization and the GPU counts. The count need to be
    # returned separately, for computing the GPU utilization when multiple DNNs
    # are running in parallel
    gpu2gputime = {g: 0 for g in plan["config"]["gpu_name_arr"]}
    gpu2count = {g: 0 for g in plan["config"]["gpu_name_arr"]}
    for pipeline_id, pipeline in enumerate(plan["pipelines"]):
        df_arr = [
            df_master[df_master.pipelineId == pipeline_id],
            *df_lb_arr_arr[pipeline_id],
        ]
        for partition, df in zip(pipeline["partitions"], df_arr):
            for _, df_ in df.groupby("workerId"):
                # Only use first request of a batch
                df_ = df_.groupby("batchId").first()
                gpu = partition["gpu"]
                mps = partition["mps"]
                total_busy = (df_.timeComplete - df_.timeStartInference).sum()
                gpu2gputime[gpu] += total_busy / (mps + 1)
                gpu2count[gpu] += 1 / (mps + 1)

    gpu2gpuutil = {}
    total_wall_time = df.arrivalTime.max() - df.arrivalTime.min()
    for gpu in plan["config"]["gpu_name_arr"]:
        if gpu2count[gpu] > 0:
            util = gpu2gputime[gpu] / gpu2count[gpu] / total_wall_time
            util = min(util, 1.0)
            gpu2gpuutil[gpu] = util
        else:
            gpu2gpuutil[gpu] = 0

    return [
        num_streams_planned,
        num_streams,
        perc_dropped,
        perc_violate_sla,
        lat_batch_build_mean,
        lat_batch_build_p99,
        lat_queue_total_mean,
        lat_queue_total_p99,
        lat_queue_mean_str,
        lat_queue_p99_str,
        max_queue_len,
        plan_slack,
        bs_planned_arr,
        bs_actual_arr,
        drop_cause_perc_arr,
        est_batch_build_lat,
        json.dumps(gpu2gpuutil),
        json.dumps(gpu2count),
    ]


def is_high_end(gpu):
    return {
        "L4": True,
        "V100": True,
        "T4": False,
        "P4": False,
    }[gpu]


def parse_cluster_sim():
    parser = argparse.ArgumentParser()
    parser.add_argument("expr",
                        choices=["multi_dnn_maf19", "multi_dnn_maf21", "ablation_maf19", "ablation_maf21"])
    parser.add_argument("-p", "--plans-dir", type=Path)
    parser.add_argument("-l", "--logs-dir", type=Path)
    parser.add_argument(
        "-d",
        "--duration",
        type=int,
        help="Duration in us of the experiment used to exclude the rampdown region",
    )
    args = parser.parse_args()

    if args.expr == "multi_dnn_maf19":
        config_fn = partial(sim_config.multitask_configs,
                            workload_weights=['0.3', '0.33', '0.37'], use_trace_arrival=False)
    elif args.expr == "multi_dnn_maf21":
        config_fn = partial(sim_config.multitask_configs,
                            workload_weights=['0.39', '0.26', '0.35'], use_trace_arrival=True)
    elif args.expr == "ablation_maf19":
        config_fn = partial(sim_config.ablation_configs, use_trace_arrival=False)
    elif args.expr == "ablation_maf21":
        config_fn = partial(sim_config.ablation_configs, use_trace_arrival=True)
    else:
        raise NotImplementedError(args.expr)

    df = []
    for config in config_fn(args.plans_dir, args.logs_dir):
        (
            plan_path,
            dnn,
            log_dir,
            xput,
            dnn_id,
            lb,
            use_trace_arrival,
            dnn_group_id,
            gpu_models,
            gpu_counts,
            bw,
            scheduler,
        ) = config

        with open(plan_path, encoding="utf-8") as f:
            plan = json.load(f)[dnn_id]

        for lf in range(100, 0, -5):
            log_path = log_dir / str(lf)
            if not log_path.exists():
                break
            results = parse_cluster_sim_ilppartitioned(
                log_path,
                plan["sla"],
                plan,
                args.duration,
            )
            df.append(
                [
                    dnn,
                    dnn_group_id,
                    dnn_id,
                    plan["sla"],
                    gpu_models,
                    gpu_counts,
                    bw,
                    xput,
                    lf,
                    scheduler,
                    *results,
                ]
            )

    columns = [
        "dnn",
        "dnn_group_id",
        "dnn_id",
        "slo",
        "gpu_models",
        "gpu_counts",
        "bw",
        "xput",
        "lf",
        "scheduler",
        "num_streams_planned",
        "num_streams",
        "perc_dropped",
        "perc_violate_sla",
        "lat_batch_build_mean",
        "lat_batch_build_p99",
        "lat_queue_total_mean",
        "lat_queue_total_p99",
        "lat_queue_mean_str",
        "lat_queue_p99_str",
        "max_queue_len",
        "plan_slack",
        "bs_planned_arr",
        "bs_actual_arr",
        "drop_cause_perc_arr",
        "est_batch_build_lat",
        "gpu2gpuutil",
        "gpu2count",
    ]
    df = pd.DataFrame(df, columns=columns)
    df.to_csv(args.logs_dir / "logs.csv", index=False)


if __name__ == "__main__":
    parse_cluster_sim()
