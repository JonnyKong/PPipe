import itertools
from dataclasses import asdict
from dataclasses import dataclass
from enum import auto
from enum import IntEnum
from pathlib import Path
from typing import List
from typing import Optional

import gurobipy as gp
import numpy as np
import pandas as pd
from contract_layers import read_layerwise_runtime
from gurobipy import GRB


class RuntimeFmt(IntEnum):
    BLOCKWISE = auto()
    LAYERWISE_W_PREPARTITION = auto()
    LAYERWISE = auto()
    LAYERWISE_W_EVEN_RUNTIME_PREPARTITION = auto()


def get_mapping_dir():
    dirs = [
        'data/prepartition_mappings',
    ]
    for d in dirs:
        if Path(d).exists():
            return Path(d)
    raise FileNotFoundError()


def get_even_runtime_prepartition_mapping_dir():
    dirs = [
        'contraction_mappings_even_runtime_part',
        '/export2/kong102/clusterserving_results/contraction_mappings_even_runtime_part',
    ]
    for d in dirs:
        if Path(d).exists():
            return Path(d)
    raise FileNotFoundError()


def get_datadir():
    return Path('./data')


def find_best_contraction_mapping_tag(dir: Path):
    """
    Search for the contraction mapping tag with the tightest upper / lower bound.
    """
    tag_arr = [
        'ilpprepart10-1.0-1.0',
        'ilpprepart10-0.7-1.3',
        'ilpprepart10-0.5-1.5',
        'ilpprepart10-0.5-2.0',
        'ilpprepart10-0.5-2.5',
        'ilpprepart10-0.5-3.0',
        'ilpprepart10-0.5-3.5',
        'ilpprepart10-0.5-4.0',
        'ilpprepart10-0.4-4.0',
    ]
    for t in tag_arr:
        if (dir / f'contraction_mapping_{t}.csv').exists():
            return t
    assert False


def read_runtime_from_blockwise_profiling(dnn, gpu, mps_id):
    df = pd.read_csv(get_blockwise_profiling_path(dnn, f'{gpu}-{mps_id + 1}'),
                     header=None)
    return df


def get_blockwise_profiling_path(dnn, gpu_mps_str):
    return get_datadir() / 'models' / 'block-timing-tf32' / f'{dnn}-{gpu_mps_str}.csv'


def read_runtime_from_layerwise_profiling_w_prepartition(dnn, gpu, mps_id):
    df = read_layerwise_runtime(get_layerwise_profiling_path(dnn, f'{gpu}-{mps_id + 1}'))

    # Contract layers
    tag = find_best_contraction_mapping_tag(get_mapping_dir() / dnn)
    df_mapping = pd.read_csv(get_mapping_dir() / dnn /
                             f'contraction_mapping_{tag}.csv',
                             header=None)
    assert df_mapping[1].is_monotonic_increasing
    df_mapping = df_mapping.rename(columns={1: 'layer_group_id'})
    # Inner join will remove the reformatting layers
    df = (
        df.merge(df_mapping, on=[0])
        .groupby(['layer_group_id'])
        .sum(numeric_only=True)
        .reset_index(drop=True)
    )
    df.columns = np.arange(len(df.columns))
    return df


def get_layerwise_profiling_path(dnn, gpu_mps_str):
    return get_datadir() / 'layer-timing-final' / dnn / gpu_mps_str / 'runtime.csv'


def read_runtime_from_layerwise_profiling(dnn, gpu, mps_id):
    p = get_layerwise_profiling_path(dnn, f'{gpu}-{mps_id + 1}')
    df = pd.read_csv(p, header=None)
    df = df.iloc[:, 1:]
    # df = df.iloc[:50, :]    # TODO: remove this
    df.columns = np.arange(len(df.columns))
    return df


def read_runtime_from_layerwise_profiling_w_even_runtime_partition(dnn, gpu, mps_id):
    df = read_layerwise_runtime(get_layerwise_profiling_path(dnn, f'{gpu}-{mps_id + 1}'))

    # Contract layers
    tag = find_best_contraction_mapping_tag(get_even_runtime_prepartition_mapping_dir() / dnn)
    df_mapping = pd.read_csv(get_even_runtime_prepartition_mapping_dir() / dnn /
                             f'contraction_mapping_{tag}.csv',
                             header=None)
    assert df_mapping[1].is_monotonic_increasing
    df_mapping = df_mapping.rename(columns={1: 'layer_group_id'})
    # Inner join will remove the reformatting layers
    df = (
        df.merge(df_mapping, on=[0])
        .groupby(['layer_group_id'])
        .sum(numeric_only=True)
        .reset_index(drop=True)
    )
    df.columns = np.arange(len(df.columns))
    return df


def get_read_runtime_fn(runtime_fmt):
    if runtime_fmt == RuntimeFmt.BLOCKWISE:
        return read_runtime_from_blockwise_profiling
    elif runtime_fmt == RuntimeFmt.LAYERWISE_W_PREPARTITION:
        return read_runtime_from_layerwise_profiling_w_prepartition
    elif runtime_fmt == RuntimeFmt.LAYERWISE:
        return read_runtime_from_layerwise_profiling
    elif runtime_fmt == RuntimeFmt.LAYERWISE_W_EVEN_RUNTIME_PREPARTITION:
        return read_runtime_from_layerwise_profiling_w_even_runtime_partition
    else:
        raise NotImplementedError()


@dataclass
class Config():
    dnn_name: str
    gpu_name_arr: List[str]
    sla: int
    num_mps_levels: List[int]
    max_num_parts: int
    bs_same: bool
    transmit_time_us_arr: List[float]
    est_xput: Optional[float]
    batch_build_factor: float
    hist_adjustment: int
    hist_adjustment_w_scheduling: int
    num_gpu_per_server_arr: List[int]
    force_sum_gpu_integer_per_partition: bool
    bw_gbps: int
    runtime_fmt: RuntimeFmt

    def num_gpu_type(self):
        return len(self.gpu_name_arr)


@dataclass
class MultitaskConfig():
    dnn_name_arr: List[str]
    gpu_limit_arr: List[int]
    gpu_name_arr: List[str]
    sla_arr: List[int]
    workload_weights: List[float]
    num_mps_levels: List[int]
    max_num_parts: int
    bs_same: bool
    # An estimated xput used to estimate the batch building latency. It could be estimated from
    # the baseline setup
    est_xput_arr: Optional[List[float]]
    batch_build_factor: float
    # Deduct from SLA in ILP, not in SLA of runtime scheduling
    hist_adjustment_arr: List[int]
    # Deduct from SLA of both ILP and runtime scheduling
    hist_adjustment_w_scheduling_arr: List[int]
    num_gpu_per_server_arr: List[int]
    force_sum_gpu_integer_per_partition: bool
    bw_gbps: int
    runtime_fmt: RuntimeFmt

    def get_singletask_cfgs(self) -> List[Config]:
        ret = []
        # for dnn_name, sla, est_xput, hist_adjustment, hist_adjustment_w_scheduling in zip(self.dnn_name_arr, self.sla_arr, self.est_xput_arr, self.hist_adjustment_arr, self.hist_adjustment_w_scheduling_arr):
        for i in range(len(self.dnn_name_arr)):
            dnn_name = self.dnn_name_arr[i]
            sla = self.sla_arr[i]
            est_xput = self.est_xput_arr[i] if self.est_xput_arr else None
            hist_adjustment = self.hist_adjustment_arr[i]
            hist_adjustment_w_scheduling = self.hist_adjustment_w_scheduling_arr[i]

            # Parse transmission time
            tag = find_best_contraction_mapping_tag(get_mapping_dir() / dnn_name)
            if self.runtime_fmt in [RuntimeFmt.BLOCKWISE, RuntimeFmt.LAYERWISE_W_PREPARTITION, RuntimeFmt.LAYERWISE_W_EVEN_RUNTIME_PREPARTITION]:
                transmit_time_file = get_mapping_dir() / dnn_name / \
                    f'contraction_trans-size_{tag}.csv'
                transmit_time_us_arr = pd.read_csv(transmit_time_file, header=None)[0].to_numpy()
                # Bytes to 100Gbps transmit time in us
                transmit_time_us_arr = (transmit_time_us_arr * 8) / (self.bw_gbps * 1e9) * 1e6
                transmit_time_us_arr = transmit_time_us_arr.tolist()
            elif self.runtime_fmt == RuntimeFmt.LAYERWISE:
                large_int = 2000
                # Use 1.0 over 0.0 to avoid zero-division error when
                # calculating UL/DL throughput
                transmit_time_us_arr = [1.0 for _ in range(large_int)]

            ret.append(Config(
                dnn_name, self.gpu_name_arr, sla, self.num_mps_levels,
                self.max_num_parts, self.bs_same, transmit_time_us_arr, est_xput,
                self.batch_build_factor, hist_adjustment, hist_adjustment_w_scheduling,
                self.num_gpu_per_server_arr, self.force_sum_gpu_integer_per_partition,
                self.bw_gbps, self.runtime_fmt))
        return ret


class Partition():

    def __init__(self, model, cfg: Config, gpu_id, partition_table):
        self.cfg = cfg
        self.partition_table = partition_table
        self.gpu_id = gpu_id
        self.gpu_name = cfg.gpu_name_arr[gpu_id]
        self.num_gpu_per_server = cfg.num_gpu_per_server_arr[gpu_id]

        # Create 4D table of dimensions (num_mps, num_bs, num_layers, num_layers) of
        # decision variables
        self.lat_table_arr_arr = []
        self.xput_table_arr_arr = []
        self.config_mat_arr_arr = []
        self.num_gpu_mat_arr_arr = []
        for mps_id in range(cfg.num_mps_levels[gpu_id]):
            read_runtime_fn = get_read_runtime_fn(cfg.runtime_fmt)
            df = read_runtime_fn(cfg.dnn_name, self.gpu_name, mps_id)
            max_bs = len(df.columns)

            # num_layers might be smaller than the specified number of partitions when
            # some layer has runtime larger than total_runtime / num_layer_groups
            self.num_layers = len(df)

            lat_table_arr = []
            xput_table_arr = []
            config_mat_arr = []
            num_gpu_mat_arr = []

            for bs_id in range(0, max_bs):
                bs = bs_id + 1

                # lat_table[i, j] means latency of layers [i, j)
                lat_arr = df[bs_id].to_numpy()
                lat_arr = np.clip(lat_arr, a_min=1.0, a_max=None)   # For numerical stability
                lat_arr_cumsum = np.cumsum(lat_arr)
                lat_table = np.zeros((self.num_layers + 1, self.num_layers + 1))
                xput_table = np.zeros((self.num_layers + 1, self.num_layers + 1))
                for i in range(self.num_layers + 1):
                    for j in range(i, self.num_layers + 1):
                        rt1 = lat_arr_cumsum[i - 1] if i > 0 else 0
                        rt2 = lat_arr_cumsum[j - 1] if j > 0 else 0
                        lat_table[i, j] = rt2 - rt1

                        if j == i:
                            xput_table[i, j] = 0
                        else:
                            # Set min dl/ul time to 1 to avoid div by 0
                            dl_time = (cfg.transmit_time_us_arr[i - 1]
                                       if (i > 0 and i < self.num_layers) else 1)
                            ul_time = (cfg.transmit_time_us_arr[j - 1]
                                       if (j > 0 and j < self.num_layers) else 1)
                            xput_dl = 1e6 / dl_time / (mps_id + 1) / self.num_gpu_per_server
                            xput_ul = 1e6 / ul_time / (mps_id + 1) / self.num_gpu_per_server
                            xput_infer = 1e6 * bs / lat_table[i, j]
                            xput_table[i, j] = min(xput_infer, xput_dl, xput_ul)

                config_mat = np.array([[model.addVar(vtype=GRB.BINARY)
                                        for j in range(self.num_layers + 1)]
                                       for i in range(self.num_layers + 1)])
                num_gpu_mat = np.array([[model.addVar(vtype=GRB.INTEGER, lb=0)
                                         for j in range(self.num_layers + 1)]
                                        for i in range(self.num_layers + 1)])

                for i in range(self.num_layers + 1):
                    for j in range(self.num_layers + 1):
                        model.addConstr(
                            (partition_table[i][j] == 0) >> (config_mat[i, j] == 0))
                        model.addConstr(
                            (config_mat[i, j] == 0) >> (num_gpu_mat[i, j] == 0))

                lat_table_arr.append(lat_table)
                xput_table_arr.append(xput_table)
                config_mat_arr.append(config_mat)
                num_gpu_mat_arr.append(num_gpu_mat)

            self.lat_table_arr_arr.append(lat_table_arr)
            self.xput_table_arr_arr.append(xput_table_arr)
            self.config_mat_arr_arr.append(config_mat_arr)
            self.num_gpu_mat_arr_arr.append(num_gpu_mat_arr)

        # Symmetry breaking, enforce skipped partitions are at the end
        for i in range(partition_table.shape[0] - 1):
            model.addConstr(partition_table[i][i] == 0)

        # Represent latency and xput as decision variable expressions
        self.lat_infer = 0
        self.lat_trans = 0
        self.xput = 0
        self.num_gpu = 0
        num_diagonal_selected = 0
        num_config = 0  # Diagonal configs selected means this partition is not used
        for mps_id in range(cfg.num_mps_levels[gpu_id]):
            for bs_id in range(len(self.lat_table_arr_arr[mps_id])):
                for i in range(self.num_layers + 1):
                    for j in range(i, self.num_layers + 1):
                        self.lat_infer += self.config_mat_arr_arr[mps_id][bs_id][i, j] \
                            * self.lat_table_arr_arr[mps_id][bs_id][i, j]
                        self.xput += self.num_gpu_mat_arr_arr[mps_id][bs_id][i, j] \
                            * self.xput_table_arr_arr[mps_id][bs_id][i, j]
                        num_config += self.config_mat_arr_arr[mps_id][bs_id][i, j]
                        self.num_gpu += self.num_gpu_mat_arr_arr[mps_id][bs_id][i, j] \
                            / (mps_id + 1)
                num_diagonal_selected += np.sum([self.config_mat_arr_arr[mps_id][bs_id][i, i]
                                                 for i in range(self.num_layers + 1)])
                # Calculate transfer latency. Exclude first column because 1 in the first column
                # will not appear & is meaningless. Exclude last column because we don't care the
                # transmission time of the final result
                transmit_time_us_arr_maybe_exclude_last_cut = cfg.transmit_time_us_arr[:self.num_layers - 1]
                self.lat_trans += np.sum(np.sum(
                    self.config_mat_arr_arr[mps_id][bs_id][:, 1:-1], axis=0)
                    * transmit_time_us_arr_maybe_exclude_last_cut) * (bs_id + 1)

        model.addConstr(num_config == 1)
        self.is_skipped = model.addVar(vtype=GRB.BINARY)
        model.addConstr(self.is_skipped == num_diagonal_selected)

        # Represent BS as a one-hot vector
        max_bs = 128     # Set this to the max bs across all GPU and CPUs
        self.bs_arr = [0 for _ in range(max_bs)]
        for mps_id in range(cfg.num_mps_levels[gpu_id]):
            for bs_id in range(min(len(self.lat_table_arr_arr[mps_id]), max_bs)):
                bs = bs_id + 1
                self.bs_arr[bs - 1] += np.sum(self.config_mat_arr_arr[mps_id][bs_id])

        # Also represent bs as an int
        self.bs = model.addVar(vtype=GRB.INTEGER, lb=0)
        model.addConstr(self.bs == np.sum([self.bs_arr[i] * (i + 1)
                        for i in range(len(self.bs_arr))]))

    def __str__(self):
        # Find which layers are ran
        layer_start = -1
        layer_end = -1
        for i in range(self.num_layers + 1):
            for j in range(i, self.num_layers + 1):
                epsilon = 0.1
                if get_gp_value(self.partition_table[i][j]) > epsilon:
                    layer_start = i
                    layer_end = j

        # Find what MPS and BS are used
        bs = -1
        mps_id = -1
        for _mps_id in range(self.cfg.num_mps_levels[self.gpu_id]):
            for bs_id in range(len(self.config_mat_arr_arr[_mps_id])):
                epsilon = 0.9
                if get_gp_value(np.sum(self.config_mat_arr_arr[_mps_id][bs_id])) > epsilon:
                    bs = bs_id + 1
                    mps_id = _mps_id
                    break

        ret = f'    layers [{layer_start}, {layer_end}]'
        ret += f', gpu={self.gpu_name}, mps={mps_id}, bs={bs}, num_gpu={get_gp_value(self.num_gpu)}'
        ret += f', lat_infer={get_gp_value(self.lat_infer)}, lat_trans={get_gp_value(self.lat_trans)}'
        ret += f', xput={get_gp_value(self.xput):.02f}'
        return ret

    def serialize(self):
        # Find which layers are ran
        layer_start = -1
        layer_end = -1
        for i in range(self.num_layers + 1):
            for j in range(i, self.num_layers + 1):
                epsilon = 0.1
                if get_gp_value(self.partition_table[i][j]) > epsilon:
                    layer_start = i
                    layer_end = j

        # Find what MPS and BS are used
        bs = -1
        mps_id = -1
        for _mps_id in range(self.cfg.num_mps_levels[self.gpu_id]):
            for bs_id in range(len(self.config_mat_arr_arr[_mps_id])):
                epsilon = 0.9
                if get_gp_value(np.sum(self.config_mat_arr_arr[_mps_id][bs_id])) > epsilon:
                    bs = bs_id + 1
                    mps_id = _mps_id
        ret = {
            'dnn': self.cfg.dnn_name,
            'layers': [layer_start, layer_end],
            'gpu': self.gpu_name,
            'mps': mps_id,
            'bs': bs,
            'num_gpu_per_server': self.num_gpu_per_server,
            'num_gpu': get_gp_value(self.num_gpu),
            'lat_infer': get_gp_value(self.lat_infer),
            'lat_trans': get_gp_value(self.lat_trans),
            'xput': get_gp_value(self.xput),
        }
        return ret


class Pipeline():

    def __init__(self, model, cfg: Config, gpu_assignment_arr):
        self.cfg = cfg

        num_parts = len(gpu_assignment_arr)

        # Read how many actual layer group there are
        # num_layers might be smaller than the specified number of partitions when
        # some layer has runtime larger than total_runtime / num_layer_groups
        read_runtime_fn = get_read_runtime_fn(cfg.runtime_fmt)
        df = read_runtime_fn(cfg.dnn_name, 'L4', 0)
        num_layers = len(df)

        # Create a partition table for each pipeline partition, where table[i][j][k]=1 means
        # pipeline partition i runs DNN model layers [j, k)
        self.partition_table_arr = []
        for part_id in range(num_parts):
            t = np.array([[model.addVar(vtype=GRB.BINARY)
                         for i in range(num_layers + 1)]
                          for j in range(num_layers + 1)])
            # Enforce lower left corner is all 0
            for i in range(1, num_layers + 1):
                for j in range(i):
                    model.addConstr(t[i, j] == 0)
            model.addConstr(np.sum(t) == 1)
            self.partition_table_arr.append(t)

        # First partition includes the first layer
        model.addConstr(np.sum(self.partition_table_arr[0][0, :]) == 1)
        # Last partition includes the last layer
        model.addConstr(np.sum(self.partition_table_arr[-1][:, -1]) == 1)
        # With no gap in between
        for part_id in range(num_parts - 1):
            for layer_id in range(num_layers + 1):
                part_ends_at_this_layer_exclusive = model.addVar(vtype=GRB.BINARY)
                model.addConstr(
                    part_ends_at_this_layer_exclusive ==
                    (gp.or_(self.partition_table_arr[part_id][:, layer_id].tolist())))
                model.addConstr(
                    (part_ends_at_this_layer_exclusive == 1) >>
                    (np.sum(self.partition_table_arr[part_id + 1][layer_id, :]) == 1))

        self.part_arr = [Partition(model, cfg, gpu_assignment_arr[i], self.partition_table_arr[i])
                         for i in range(num_parts)]

        if cfg.bs_same:
            # Assumes the first part is not skipped due to symmetry breaking
            for part_id in range(1, num_parts):
                model.addConstr((self.part_arr[part_id].is_skipped == 0)
                                >> (self.part_arr[part_id].bs == self.part_arr[0].bs))

        # Represent xput as auxiliary variable
        self.xput = model.addVar(vtype=GRB.CONTINUOUS)
        for p in self.part_arr:
            model.addConstr((p.is_skipped == 0) >> (self.xput <= p.xput))

        # Num gpu per partition, per gpu type
        # self.num_gpu_mat = np.zeros((num_parts, len(cfg.gpu_name_arr)))
        self.num_gpu_mat = [[0 for j in range(len(cfg.gpu_name_arr))]
                            for i in range(num_parts)]
        for i in range(len(self.part_arr)):
            self.num_gpu_mat[i][gpu_assignment_arr[i]] += self.part_arr[i].num_gpu
        self.num_gpu_mat = np.array(self.num_gpu_mat)

        # Sum up number of GPU used by type
        self.num_gpu_arr = np.sum(self.num_gpu_mat, axis=0)

        # Satisfy SLA
        self.est_batch_build_lat = 0.0
        if cfg.est_xput:
            self.est_batch_build_lat = cfg.batch_build_factor * 1e6 \
                * (self.part_arr[0].bs - 1) / cfg.est_xput
        self.lat = np.sum([p.lat_infer + p.lat_trans for p in self.part_arr])
        self.lat += self.est_batch_build_lat
        self.lat += (self.cfg.hist_adjustment + self.cfg.hist_adjustment_w_scheduling)
        # Represent SLA as auxiliary variable
        self.lat_var = model.addVar(vtype=GRB.CONTINUOUS)
        model.addConstr(self.lat_var == self.lat)
        self.is_used = model.addVar(vtype=GRB.BINARY)
        # https://support.gurobi.com/hc/en-us/community/posts/360074960552/comments/360013557551
        # - If xput=0, then is_used=0 to avoid violating the first constr
        # - If xput>0, then is_used=1 to avoid violating the second constr
        model.addConstr((self.is_used == 0) >> (np.sum(self.num_gpu_arr) == 0))
        model.addConstr((self.is_used == 1) >> (self.lat_var <= cfg.sla))

    def __str__(self):
        s = f'  Pipeline: xput={get_gp_value(self.xput):.02f}, lat={get_gp_value(self.lat):.02f}'
        s += f', est_xput: {get_gp_value(self.cfg.est_xput)}'
        s += f', est_batch_build_lat: {get_gp_value(self.est_batch_build_lat):.02f}'
        s += f', batch_build_factor: {self.cfg.batch_build_factor:.02f}'
        s += f', hist_adjustment: {self.cfg.hist_adjustment}\n'
        s += f', hist_adjustment_w_scheduling: {self.cfg.hist_adjustment_w_scheduling}\n'
        for part in self.part_arr:
            epsilon = 0.1
            if get_gp_value(part.is_skipped) < epsilon:
                s += f'{part}\n'
        return s

    def serialize(self):
        ret = {
            'xput': get_gp_value(self.xput),
            'partitions': [],
            'est_xput': self.cfg.est_xput,
            'est_batch_build_lat': get_gp_value(self.est_batch_build_lat),
            'batch_build_factor': self.cfg.batch_build_factor,
            'hist_adjustment': self.cfg.hist_adjustment,
            'hist_adjustment_w_scheduling': self.cfg.hist_adjustment_w_scheduling,
        }
        for part in self.part_arr:
            epsilon = 0.1
            if get_gp_value(part.is_skipped) < epsilon:
                ret['partitions'].append(part.serialize())
        return ret


class Cluster():

    def __init__(self, model, cfg: Config):
        self.cfg = cfg

        # Enumerate all possible combination of pipelines
        gpu_assignment_arr_arr = list(itertools.product(
            range(len(cfg.gpu_name_arr)), repeat=cfg.max_num_parts))

        self.pipeline_arr = [Pipeline(model, cfg, n)
                             for n in gpu_assignment_arr_arr]

        # Sum up number of GPU used by type
        if not cfg.force_sum_gpu_integer_per_partition:
            self.num_gpu_arr = np.sum(np.array([p.num_gpu_arr for p in self.pipeline_arr]), axis=0)
        else:
            # [num_parts, num_gpu_type]
            self.num_gpu_mat_ceil = np.array([[model.addVar(vtype=GRB.INTEGER, lb=0)
                                               for j in range(len(cfg.gpu_name_arr))]
                                              for i in range(cfg.max_num_parts)])
            self.num_gpu_mat = [[0
                                 for j in range(len(cfg.gpu_name_arr))]
                                for i in range(cfg.max_num_parts)]
            for i in range(cfg.max_num_parts):
                for j in range(len(cfg.gpu_name_arr)):
                    self.num_gpu_mat[i][j] = np.sum([p.num_gpu_mat[i, j]
                                                    for p in self.pipeline_arr])
                    model.addConstr(self.num_gpu_mat_ceil[i, j] >= self.num_gpu_mat[i][j])
            self.num_gpu_mat = np.array(self.num_gpu_mat)
            self.num_gpu_arr = np.sum(self.num_gpu_mat, axis=0)
            self.num_gpu_arr_ceil = np.sum(self.num_gpu_mat_ceil, axis=0)
        self.num_gpu = np.sum(self.num_gpu_arr)

        self.xput = np.sum([p.xput for p in self.pipeline_arr])

    def __str__(self):
        # Type will be gp.Var if only one pipeline
        xput = get_gp_value(self.xput)
        num_gpu_str = f'{get_gp_value(self.num_gpu):.02f}'

        s = f'Cluster: xput={xput:.02f}, num_gpu={num_gpu_str}, bw_gbps={self.cfg.bw_gbps}\n'
        for pipeline in self.pipeline_arr:
            epsilon = 0.1
            if not pipeline.xput.X > epsilon:
                continue
            s += f'{pipeline}'
        return s

    def serialize(self):
        xput = get_gp_value(self.xput)
        ret = {
            'xput': xput,
            'config': asdict(self.cfg),
            'sla': self.cfg.sla,    # Exists for backward compatibility
            'pipelines': []
        }
        for pipeline in self.pipeline_arr:
            epsilon = 0.1
            if not pipeline.xput.X > epsilon:
                continue
            ret['pipelines'].append(pipeline.serialize())
        return ret


class MultitaskCluster():

    def __init__(self, model, multitask_cfg: MultitaskConfig):
        self.multitask_cfg = multitask_cfg

        self.cluster_arr = []
        for cfg in multitask_cfg.get_singletask_cfgs():
            self.cluster_arr.append(Cluster(model, cfg))

        # GPU count constraint
        if not multitask_cfg.force_sum_gpu_integer_per_partition:
            self.num_gpu_arr = np.sum(np.array([c.num_gpu_arr for c in self.cluster_arr]), axis=0)
        else:
            self.num_gpu_arr = np.sum(
                np.array([c.num_gpu_arr_ceil for c in self.cluster_arr]), axis=0)
        self.num_gpu = np.sum(self.num_gpu_arr)
        for i in range(len(multitask_cfg.gpu_name_arr)):
            model.addConstr(self.num_gpu_arr[i] <= multitask_cfg.gpu_limit_arr[i])

        # Maximize the min throughput, normalized by the workload weights
        self.xput = model.addVar(vtype=GRB.CONTINUOUS)
        for i, c in enumerate(self.cluster_arr):
            model.addConstr(self.xput <= c.xput / multitask_cfg.workload_weights[i])

    def __str__(self):
        ret = ''
        for i in range(len(self.multitask_cfg.dnn_name_arr)):
            ret += f'{self.multitask_cfg.dnn_name_arr[i]}\n'
            ret += str(self.cluster_arr[i])
        return ret

    def serialize(self):
        return list(c.serialize() for c in self.cluster_arr)


def get_gp_value(v):
    if isinstance(v, gp.Var):
        return v.X
    elif isinstance(v, gp.LinExpr):
        return v.getValue()
    else:
        return v
