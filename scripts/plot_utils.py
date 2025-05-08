import numpy as np


def pad_attainments_to_full_range_of_load_factor(load_factors, attainments, load_factors_full):
    # Make sure existing parts match
    assert load_factors == load_factors_full[(-1 * len(load_factors)):]

    num_to_pad = len(load_factors_full) - len(attainments)
    attainments = np.concatenate([[attainments[0]] * num_to_pad, attainments])
    return attainments.tolist()


def plot_grouped_bar(ax, x_arr_arr, label_arr, xtick_labels, x_stdev_arr_arr=None,
                     plot_bar_label: bool = False, color_arr=None, hatch_arr=None):
    if color_arr is None:
        color_arr = [f'C{i}' for i in range(len(label_arr))]
    x_arr_arr = np.array(x_arr_arr)
    assert x_arr_arr.shape[0] == len(label_arr)
    if x_stdev_arr_arr:
        x_stdev_arr_arr = np.array(x_stdev_arr_arr)

    x = np.arange(x_arr_arr.shape[1])  # the label locations
    width = 0.75 / x_arr_arr.shape[0]  # the width of the bars

    for i in range(len(label_arr)):
        offset = width * i
        rects = ax.bar(x + offset, x_arr_arr[i, ...], width, label=label_arr[i],
                       edgecolor='black', color=color_arr[i],
                       hatch=(hatch_arr[i] if hatch_arr else None))
        if x_stdev_arr_arr is not None:
            ax.errorbar(x + offset, x_arr_arr[i, ...], yerr=x_stdev_arr_arr[i, ...],
                        fmt='none', color='k', capsize=2)
        if plot_bar_label:
            ax.bar_label(rects, padding=1, fmt='%.2f')
    xticks = x + width * (x_arr_arr.shape[0] - 1) / 2
    ax.set_xticks(xticks, xtick_labels)

def filter_largest_load_factor_each_setting_w_99_attainment(df, settings):
    def get_largest_lf_run(group_df):
        sorted_group = group_df.sort_values(by='lf', ascending=False)
        # Filter runs with attainment larger than 99%
        larger = sorted_group[sorted_group['attainment'] > 99]
        if not larger.empty:
            return larger.iloc[0]  # Return the first larger value as a row

        # If non larger than 99, simply return the smallest lf run
        return sorted_group.iloc[-1]

    ret = df.groupby(settings).apply(get_largest_lf_run)
    ret = ret.reset_index(drop=True)
    return ret


def prettify_gpus_and_counts(gpus, counts, sep=", "):
    if gpus == ["L4", "P4"]:
        if counts == [25, 75]:
            return r"HC1-L"
        else:
            return r"HC1-S"
    elif gpus == ["L4", "T4"]:
        if counts == [25, 75]:
            return r"HC2-L"
        else:
            return r"HC2-S"
    elif gpus == ["V100", "P4"]:
        if counts == [25, 75]:
            return r"HC3-L"
        else:
            return r"HC3-S"
    elif gpus == ["V100", "T4"]:
        if counts == [25, 75]:
            return r"HC4-L"
        else:
            return r"HC4-S"


def prettify_scheduler_str(s):
    return {
        "bl": "NP",
        "dart-r": "DART-r",
        "v4": "PPIPE",
        "nexus": "Reactive",
    }[s]


def scheduler2color(s):
    return {
        "bl": "C2",
        "dart-r": "C0",
        "v4": "orange",
        "nexus": "grey",
    }[s]


def scheduler2hatch(s):
    return {
        "bl": None,
        "dart-r": "/",
        "v4": "\\",
        "nexus": None,
    }[s]
