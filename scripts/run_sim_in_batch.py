import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

from sim_config import multitask_configs, ablation_configs
from parse_cluster_sim import parse_cluster_sim_ilppartitioned

SIM_DIR = Path(__file__).parent.parent / "cluster-sim"


def main(args):
    if args.expr == "multi_dnn_maf19":
        config_fn = partial(multitask_configs,
                            workload_weights=['0.3', '0.33', '0.37'], use_trace_arrival=False)
    elif args.expr == "multi_dnn_maf21":
        config_fn = partial(multitask_configs,
                            workload_weights=['0.39', '0.26', '0.35'], use_trace_arrival=True)
    elif args.expr == "ablation_maf19":
        config_fn = partial(ablation_configs, use_trace_arrival=False)
    elif args.expr == "ablation_maf21":
        config_fn = partial(ablation_configs, use_trace_arrival=True)
    else:
        raise NotImplementedError(args.expr)

    with ProcessPoolExecutor(max_workers=8) as e:
        for i, config in enumerate(config_fn(args.plan_root, args.log_dir)):
            if i % args.num_hosts == args.rank:
                cmd = [
                    SIM_DIR,
                    args.latency_root,
                    args.mapping_root,
                    *config[:7],
                    True,
                ]
                e.submit(run, *cmd)
                # run(*cmd)


def run(
    sim_dir,
    latency_root,
    mapping_root,
    plan_path,
    dnn_name,
    logdir,
    xput,
    dnn_id,
    lb,
    use_trace,
    vary_load,
):
    with open(plan_path) as f:
        plan = json.load(f)[0]

    def run_once(lf=100):
        cmd = f"{sim_dir}/build/install/cluster-sim/bin/cluster-sim \
                ilp_partitioned --dnn-name {dnn_name} \
                --latency-root={latency_root} \
                --mapping-root={mapping_root} \
                --json-plan-path={plan_path} \
                --logdir={logdir / str(lf)} \
                --load={lf * xput // 100} \
                --dnn-id={dnn_id} \
                --lb={lb}"
        if use_trace:
            trace_path = '/export2/kong102/clusterserving_results/maf_traces/azure_functions_trace_2021/AzureFunctionsInvocationTraceForTwoWeeksJan2021.txt'
            cmd += f' --trace-path={trace_path}'
        os.system(cmd)

    if vary_load:
        for lf in range(100, 0, -5):
            run_once(lf)
            metric_arr = parse_cluster_sim_ilppartitioned(
                logdir / str(lf), int(plan["sla"]), plan
            )
            perc_dropped = metric_arr[2] + metric_arr[3]
            print(f"{dnn_name} lf:{lf}, perc_dropped:{perc_dropped}")
            if perc_dropped <= 0.01:
                break
    else:
        run_once(xput)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("expr",
                        choices=["multi_dnn_maf19", "multi_dnn_maf21", "ablation_maf19", "ablation_maf21"])
    parser.add_argument("latency_root")
    parser.add_argument("mapping_root")
    parser.add_argument("plan_root", type=Path)
    parser.add_argument("log_dir", type=Path)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--num_hosts", type=int, default=1)
    args = parser.parse_args()

    main(args)
