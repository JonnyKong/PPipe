import itertools
from pathlib import Path
import numpy as np
import json
import random
from matplotlib import pyplot as plt

import pandas as pd
from .plot_for_paper import get_cdf_data

RESULT_ROOT = Path("/export2/kong102/clusterserving_results")


def random_sample_grouping(objects, group_size, n_samples):
    """
    Randomly sample groupings of `objects` into groups each of size
    `group_size`.
    """
    objects = sorted(objects)
    assert len(objects) % group_size == 0

    grouping_arr = []

    for _ in range(n_samples):
        objects_shuffled = objects
        random.shuffle(objects_shuffled)
        g = []
        for i in range(0, len(objects), group_size):
            g.append(tuple(sorted(objects_shuffled[i : i + group_size])))
        grouping_arr.append(g)
    return grouping_arr


def build_impr_table(dnn_name_arr, group_size):
    """
    Build a mapping where the key is a group of DNNs, and the key is the
    list of improvment over HCs.
    """
    dnn_name_arr = sorted(dnn_name_arr)
    dnn_name_arr_arr = list(itertools.combinations(sorted(dnn_name_arr), group_size))

    plan_dir = RESULT_ROOT / "plans_5xL4sla_block-timing" / "tf32sla-3dnns_padding-0.4"
    workload_weights = [1.0 for _ in range(group_size)]

    clusters = [
        [["V100", "T4"], [25, 75], True, 2, [4, 4], [4, 2], 10],
        [["V100", "P4"], [25, 75], True, 2, [4, 1], [2, 1], 6],
        [["L4", "T4"], [25, 75], True, 2, [4, 4], [4, 2], 10],
        [["L4", "P4"], [25, 75], True, 2, [4, 1], [1, 1], 6],
    ]

    ret = {}

    for dnn_name_arr in dnn_name_arr_arr:
        impr_arr = []

        for cluster in clusters:
            gpu_name_arr, gpu_limit_arr, _, _, _, _, bw_gbps = cluster

            tag = "-".join(dnn_name_arr)
            tag += f'_{"-".join([str(w) for w in workload_weights])}'
            tag += f'_{"-".join(gpu_name_arr)}'
            tag += f'_{"-".join([str(s) for s in gpu_limit_arr])}'
            tag += f"_bw-gbps-{bw_gbps}"

            xput_v4 = read_xput_from_plan(plan_dir / f"{tag}_sla-multiplier-5_v4.json")
            xput_bl = read_xput_from_plan(plan_dir / f"{tag}_sla-multiplier-5_bl.json")
            impr = (xput_v4 - xput_bl) / xput_bl
            impr_arr.append(impr)

        ret[dnn_name_arr] = impr_arr
    return ret


def read_xput_from_plan(plan_path):
    with open(plan_path) as f:
        plan = json.load(f)
    xputs = [p["xput"] for p in plan]
    return np.min(xputs)


def plot_impr_cdf(dnn_name_arr, group_size, impr_table):
    """
    Plot the distribution of improvement.
    """
    dnn_name_arr = sorted(dnn_name_arr)

    impr_table = build_impr_table(dnn_name_arr, group_size)

    # Randomly sample some groupings, plot impr distribution
    grouping_arr = random_sample_grouping(dnn_name_arr, group_size, 1000)
    impr_arr = []
    for grouping in grouping_arr:
        impr = np.mean([np.mean(impr_table[dnns]) for dnns in grouping])
        impr_arr.append(impr)

    fig, ax = plt.subplots(1, 1)
    x, y = get_cdf_data(impr_arr)
    ax.plot(x, y)
    ax.set_xlabel("Impr")
    ax.set_ylabel("CDF")
    fig.tight_layout()
    plt.savefig("figs_nsdi/dnn_grouping_impr_dist.png")


def get_random_grouping(dnn_name_arr, group_size, seed=0):
    assert len(dnn_name_arr) % group_size == 0

    dnn_name_arr = sorted(dnn_name_arr)
    random.seed(seed)
    random.shuffle(dnn_name_arr)

    grouping = []
    for i in range(0, len(dnn_name_arr), group_size):
        grouping.append(tuple(sorted(dnn_name_arr[i : i + group_size])))
    return grouping


if __name__ == "__main__":
    dnn_name_arr = pd.read_csv(
        "scripts/trt/model-zoo/selected_models.txt", sep=" ", header=None
    )[0].to_list()
    group_size = 3

    impr_table = build_impr_table(dnn_name_arr, group_size)
    plot_impr_cdf(dnn_name_arr, group_size, impr_table)

    # Print the improvements under different seeds, pick one with smaller
    # variance so it looks good in the paper
    for i in range(20):
        dnn_name_arr_arr = get_random_grouping(dnn_name_arr, group_size, seed=i)
        impr_arr_arr = np.array(
            [impr_table[dnn_name_arr] for dnn_name_arr in dnn_name_arr_arr]
        )
        print(i, "min: ", np.min(impr_arr_arr))
        print(impr_arr_arr)
