import random


def get_random_grouping(dnn_name_arr, group_size, seed=0):
    assert len(dnn_name_arr) % group_size == 0

    dnn_name_arr = sorted(dnn_name_arr)
    random.seed(seed)
    random.shuffle(dnn_name_arr)

    grouping = []
    for i in range(0, len(dnn_name_arr), group_size):
        grouping.append(tuple(sorted(dnn_name_arr[i: i + group_size])))
    return grouping
