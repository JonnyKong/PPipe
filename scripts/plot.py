from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import fire

from .plot_for_paper import pad_attainments_to_full_range_of_load_factor
from .plot_for_paper import pdfcrop
from .plot_for_paper import plot_grouped_bar
from .plot_for_paper_nsdi25 import filter_largest_load_factor_each_setting_w_99_attainment
from .plot_for_paper_nsdi25 import prettify_gpus_and_counts
from .plot_for_paper_nsdi25 import prettify_scheduler_str
from .plot_for_paper_nsdi25 import scheduler2color
from .plot_for_paper_nsdi25 import scheduler2hatch
from .print_cluster_sim_xfer_time import print_cluster_sim_xfer_time

RESULT_ROOT = Path("/export2/kong102/clusterserving_results")


def read_main_results():
    df_maf19 = pd.read_csv(
        RESULT_ROOT / "cluster-logs/tf32sla-3dnns_padding-0.4_workload-weights-0.30-0.33-0.37_maf19/logs.csv")
    df_maf19['trace'] = 'MAF19'

    df_maf21 = pd.read_csv(
        RESULT_ROOT / "cluster-logs/tf32sla-3dnns_padding-0.4_workload-weights-0.39-0.26-0.35_maf21/logs.csv")
    df_maf21['trace'] = 'MAF21'

    df = pd.concat([df_maf19, df_maf21])

    df["attainment"] = (1 - df.perc_dropped) * (1 - df.perc_violate_sla) * 100.0
    return df


def plot_main_results_gain_barplot(verbose: bool = False):
    df = read_main_results()
    df = filter_largest_load_factor_each_setting_w_99_attainment(
        df, settings=["dnn", "gpu_models", "gpu_counts", "scheduler", "trace"]
    )

    gpus_and_counts = [
        [["L4", "P4"], [25, 75]],
        [["L4", "T4"], [25, 75]],
        [["V100", "P4"], [25, 75]],
        [["V100", "T4"], [25, 75]],
    ]
    schedulers = ["bl", "dart-r", "v4"]

    # One figure for each HC
    fig, axs = plt.subplots(2, 4, figsize=(12, 3.5), sharey=True)

    for trace_id, trace in enumerate(['MAF19', 'MAF21']):
        for hc_id, (gpus, counts) in enumerate(gpus_and_counts):
            ax = axs[trace_id][hc_id]
            lf_arr_arr = []
            lf_stdev_arr_arr = []

            for s in schedulers:
                lf_arr = []
                lf_stdev_arr = []

                for dnn_group_id in np.arange(6):
                    df_ = df[
                        (df.gpu_models == str(gpus))
                        & (df.gpu_counts == str(counts))
                        & (df.scheduler == s)
                        & (df.dnn_group_id == dnn_group_id)
                        & (df.trace == trace)
                    ]
                    lf_arr.append(df_.lf.mean() / 100.0)
                    lf_stdev_arr.append(df_.lf.std() / 100.0)
                lf_arr_arr.append(lf_arr)
                lf_stdev_arr_arr.append(lf_stdev_arr)

            if verbose:
                print(
                    f"trace {trace}, lf gain v4 over bl: ",
                    np.array(lf_arr_arr[2]) / np.array(lf_arr_arr[0]) - 1.0,
                )
                print(
                    f"trace {trace}, lf gain v4 over dart: ",
                    np.array(lf_arr_arr[2]) / np.array(lf_arr_arr[1]) - 1.0,
                )
                print(
                    f"trace {trace}, dart gain v4 over bl: ",
                    np.array(lf_arr_arr[1]) / np.array(lf_arr_arr[0]) - 1.0,
                )
            xtick_labels = [f"G{i}" for i in np.arange(1, 7)]
            plot_grouped_bar(
                ax,
                lf_arr_arr,
                label_arr=[prettify_scheduler_str(s) for s in schedulers],
                xtick_labels=xtick_labels,
                x_stdev_arr_arr=lf_stdev_arr_arr,
                color_arr=[scheduler2color(s) for s in schedulers],
                hatch_arr=[scheduler2hatch(s) for s in schedulers],
            )

            ax.set_ylim(0, 1.05)
            if trace_id == 0:
                ax.set_title(prettify_gpus_and_counts(gpus, counts))
            if trace_id == 1:
                ax.set_xlabel("DNN groups")
            if hc_id == 0:
                ax.set_ylabel(f"Load Factor at\n99% Attainment\n({prettify_trace_str(trace)})")
            ax.grid(axis="y")
            ax.set_axisbelow(True)

    handles, labels = axs[0][0].get_legend_handles_labels()
    lgd = fig.legend(handles, labels, ncol=3, bbox_to_anchor=[0.53, 1.01], loc="center")
    fig.tight_layout()

    savepath = "figs_atc25/main_results_gain_barplot.pdf"
    plt.savefig(savepath, bbox_extra_artists=(lgd,), bbox_inches="tight")
    pdfcrop(savepath)

    if verbose:
        for trace in ['MAF19', 'MAF21']:
            df_ = df[df.trace == trace]
            print(f"Average load factor per scheduler (trace={trace}):")
            for s in schedulers:
                lf_arr = df_[df_.scheduler == s].lf
                print(f">>> {s}: mean {lf_arr.mean()}, min {lf_arr.min()}, max {lf_arr.max()}")

            print("v4 relative gain over others: ")
            lf_arr_v4 = df_[df_.scheduler == "v4"].sort_values(by=["dnn"]).lf.to_numpy()
            for s in ["bl", "dart-r"]:
                lf_arr_s = df_[df_.scheduler == s].sort_values(by=["dnn"]).lf.to_numpy()
                gain_arr = lf_arr_v4 / lf_arr_s - 1.0
                print(
                    f">>> mean gain over {s}: mean {np.mean(gain_arr)}, min {np.min(gain_arr)}, max {np.max(gain_arr)}"
                )

                print(f"trace={trace}, v4 gain over {s} over each HC")
                for gpus, _ in gpus_and_counts:
                    lf_arr_v4_per_gp = (
                        df_[(df_.scheduler == "v4") & (df_.gpu_models == str(gpus))]
                        .sort_values(by=["dnn"])
                        .lf.to_numpy()
                    )
                    lf_arr_other = (
                        df_[(df_.scheduler == s) & (df_.gpu_models == str(gpus))]
                        .sort_values(by=["dnn"])
                        .lf.to_numpy()
                    )
                    print(f">>> {np.mean(lf_arr_v4_per_gp / lf_arr_other) - 1.0}")


def plot_main_results_attainment_curve():
    """
    Four figures, one figure for one HC, for a particular DNN group.
    """
    df = read_main_results()
    df = df[df.trace == 'MAF19']
    gpus_and_counts = [
        [["L4", "P4"], [25, 75]],
        [["L4", "T4"], [25, 75]],
        [["V100", "P4"], [25, 75]],
        [["V100", "T4"], [25, 75]],
    ]
    df = df[df.dnn_group_id == 0]
    load_factor_full = [round(a, 3) for a in np.arange(0.05, 1.05, 0.05).tolist()]

    fig, axs = plt.subplots(1, len(gpus_and_counts), figsize=(12, 2.0), squeeze=False)
    schedulers = ["bl", "dart-r", "v4"]

    for i, (gpus, gpu_counts) in enumerate(gpus_and_counts):
        ax = axs[0, i]

        for j, s in enumerate(schedulers):
            df_ = df[(df.scheduler == s) & (df.gpu_models == str(gpus))]
            df_ = df_.sort_values(by="lf")

            # Pad attainment to full ranges for each DNN
            y_arr = []
            for dnn in df_.dnn.unique().tolist():
                x = (df_[df_.dnn == dnn].lf / 100.0).tolist()
                y = df_[df_.dnn == dnn].attainment.tolist()
                y = pad_attainments_to_full_range_of_load_factor(x, y, load_factor_full)
                y_arr.append(y)
            y = np.array(y_arr).mean(axis=0).tolist()

            ax.plot(
                load_factor_full,
                y,
                label=prettify_scheduler_str(s),
                linestyle="--",
                linewidth=1.5,
                marker="o",
                markersize=4,
                color=scheduler2color(s),
            )
            best_load = load_factor_full[np.where(np.array(y) >= 99)[0][-1]]
            ax.axvline(best_load, color=scheduler2color(s), linestyle=":")

        ax.set_xlabel('Load Factor')
        if i == 0:
            ax.set_ylabel(f'SLO Att.(%)')
        else:
            ax.set_yticklabels([])
        ax.set_title(prettify_gpus_and_counts(gpus, gpu_counts))
        ax.set_ylim(60, 100)
        ax.set_xlim(0, 1)

    handles, labels = axs[0, 0].get_legend_handles_labels()
    lgd = fig.legend(handles, labels, ncol=5, bbox_to_anchor=[0.5, 1.01], loc="center")
    fig.tight_layout()
    savepath = "figs_atc25/main_results_attainment_curve.pdf"
    # https://stackoverflow.com/a/10154763/6060420
    plt.savefig(savepath, bbox_extra_artists=(lgd,), bbox_inches="tight")
    pdfcrop(savepath)


def plot_ablation_nexus_barplot():
    """
    Load factor under different ILP paddings:
        +---------+------------+---------------+----------------------------+
        | Padding | Mean LF BL | Mean LF Nexus |        Mean LF Ours        |
        +---------+------------+---------------+----------------------------+
        |     0.1 |      85.28 |         70.83 | 91.39 <- will use this one |
        |     0.2 |      84.44 |         82.50 | 91.67                      |
        |    0.25 |      81.95 |         94.17 | 98.33                      |
        |     0.3 |      76.39 |         95.83 | 99.72                      |
        |     0.4 |      72.50 |         92.22 | 96.39                      |
        +---------+------------+---------------+----------------------------+
    and will not show BL due to low performance at 0.1 padding.
    """
    result_dir = Path(
        "/export2/kong102/clusterserving_results/cluster-logs/ablation-nexus_padding-0.1_maf19"
    )
    df = pd.read_csv(result_dir / "logs.csv")
    df["attainment"] = (1 - df.perc_dropped) * (1 - df.perc_violate_sla) * 100.0
    df = filter_largest_load_factor_each_setting_w_99_attainment(
        df, settings=["dnn", "gpu_models", "gpu_counts", "scheduler"]
    )

    fig, ax = plt.subplots(1, 1, figsize=(2, 2.5))

    scheduler_arr = ["nexus", "v4"]
    lf_arr = [df[(df.scheduler == s)].lf.mean() / 100.0 for s in scheduler_arr]
    lf_std_arr = [df[(df.scheduler == s)].lf.std() / 100.0 for s in scheduler_arr]
    color_arr = [scheduler2color(s) for s in scheduler_arr]
    hatch_arr = [scheduler2hatch(s) for s in scheduler_arr]
    ax.bar(
        scheduler_arr,
        lf_arr,
        color=color_arr,
        yerr=lf_std_arr,
        hatch=hatch_arr,
        capsize=2,
        edgecolor="black",
        width=0.4,
    )
    ax.set_xticklabels([prettify_scheduler_str(s) for s in scheduler_arr])
    ax.set_xlabel("Scheduler")
    ax.set_ylabel("Mean Load Factor\nat 99% Attainment")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(-0.4, 1.4)
    fig.tight_layout()

    print(f'Ablation study load factors for {scheduler_arr}: {lf_arr}')

    savepath = "figs_atc25/ablation_nexus.pdf"
    plt.savefig(savepath)
    pdfcrop(savepath)


def plot_main_results_gpu_temporal_util_barplot():
    df = read_main_results()
    df = df[df.trace == 'MAF19']

    df = filter_largest_load_factor_each_setting_w_99_attainment(
        df, settings=["dnn", "gpu_models", "gpu_counts", "scheduler"]
    )
    gpus_and_counts = [
        [["L4", "P4"], [25, 75]],
        [["L4", "T4"], [25, 75]],
        [["V100", "P4"], [25, 75]],
        [["V100", "T4"], [25, 75]],
    ]
    schedulers = ["bl", "dart-r", "v4"]
    util_arr_arr = []
    util_stdev_arr_arr = []
    low_cls_util_arr_arr = []

    for s in schedulers:
        low_cls_util_arr = []

        for gpu_class_id, gpu_class in enumerate(["High", "Low"]):
            util_arr = []
            util_stdev_arr = []
            for gpus, counts in gpus_and_counts:
                gpu = gpus[gpu_class_id]
                count = counts[gpu_class_id]

                df_ = df[
                    (df.gpu_models == str(gpus))
                    & (df.gpu_counts == str(counts))
                    & (df.scheduler == s)
                ]
                utils = []
                # Compute a util for each group
                for _, gp in df_.groupby("dnn_group_id"):
                    gpuutil_arr = [eval(row)[gpu] for row in gp.gpu2gpuutil]
                    count_arr = [eval(row)[gpu] for row in gp.gpu2count]
                    # Util is weighted by gpu count. Divide by `count` instead
                    # of `sum(count_arr)` because there may be unused GPUs
                    util = np.sum(np.array(gpuutil_arr) * np.array(count_arr)) / count
                    utils.append(util)

                util_arr.append(np.mean(utils) * 100.0)
                util_stdev_arr.append(np.std(utils) * 100.0)
            util_arr_arr.append(util_arr)
            util_stdev_arr_arr.append(util_stdev_arr)

            if gpu_class == "Low":
                low_cls_util_arr.extend(util_arr)
        low_cls_util_arr_arr.append(low_cls_util_arr)

    print("Low-class GPU utils:")
    for i, s in enumerate(schedulers):
        print(f">> {s}: {np.mean(low_cls_util_arr_arr[i])}")

    fig, ax = plt.subplots(1, 1, figsize=(5, 2.5))
    xtick_labels = [prettify_gpus_and_counts(g, c, "\n") for g, c in gpus_and_counts]
    label_arr = []
    color_arr = []
    hatch_arr = []
    for s in schedulers:
        for gpu_class in ["high", "low"]:
            label_arr.append(f"{prettify_scheduler_str(s)}, {gpu_class}-class")
            color_arr.append(scheduler2color(s))
            hatch_arr.append(None if gpu_class == "high" else "//")

    plot_grouped_bar(
        ax,
        util_arr_arr,
        label_arr=label_arr,
        xtick_labels=xtick_labels,
        x_stdev_arr_arr=util_stdev_arr_arr,
        color_arr=color_arr,
        hatch_arr=hatch_arr,
    )

    ax.set_ylabel("GPU Utilization (%)")
    ax.set_ylim(0, 100)
    ax.set_xlabel("Heterogeneous Cluster Configuration")
    ax.grid(axis="y")
    ax.set_axisbelow(True)

    handles, labels = ax.get_legend_handles_labels()
    lgd = fig.legend(handles, labels, ncol=3, bbox_to_anchor=[0.5, 1.05], loc="center")
    fig.tight_layout()

    savepath = "figs_atc25/main_results_gpu_temporal_util_barplot.pdf"
    plt.savefig(savepath, bbox_extra_artists=(lgd,), bbox_inches="tight")
    pdfcrop(savepath)


def print_ablation_study_transfer_time():
    full_bs = {
        'v4': 0,
        'nexus': 0,
    }
    # plans_5xL4sla_block-timing/tf32sla_padding-0.10_max-num-parts-3/efficientnet-b8_3rdparty_8xb32-aa-advprop_in1k_L4-T4_25-75_bw-gbps-10_sla-multiplier-5_v4.json
    full_bs = 2

    for s in ['nexus', 'v4']:
        cluster_log = f'/export2/kong102/clusterserving_results/cluster-logs/ablation-nexus_padding-0.1_maf19/efficientnet-b8_3rdparty_8xb32-aa-advprop_in1k_L4-T4_25-75_bw-gbps-10_sla-multiplier-5_{s}/dnn-id-0/100/1-0.csv'

        print(f'Transfer time for {s}:')
        print_cluster_sim_xfer_time(cluster_log, full_bs)


def prettify_trace_str(s):
    return {
        'MAF19': 'Poisson',
        'MAF21': 'Bursty',
    }[s]


if __name__ == "__main__":
    fire.Fire({
        'fig6': plot_main_results_gain_barplot,
        'fig7': plot_main_results_attainment_curve,
        'fig8': plot_main_results_gpu_temporal_util_barplot,
    })
    # plot_main_results_gain_barplot(verbose=True)
    # plot_main_results_attainment_curve()
    # plot_ablation_nexus_barplot()
    # plot_main_results_gpu_temporal_util_barplot()
    # print_ablation_study_transfer_time()
