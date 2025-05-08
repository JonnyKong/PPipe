"""
Pre-partition into N pieces with similar latency (at a given batch size)
while minimizing the amount of data transmission between the pieces.

@Author kong102@purdue.edu
@Date 2023-03-12
"""
import json
import sys
import time
from pathlib import Path

import gurobipy as gp
import numpy as np
import pandas as pd
from contract_layers import read_layerwise_runtime
from gurobipy import GRB

gpu_name_arr = ["V100", "L4", "T4", "P4"]
num_partitions = 10


class DNNModel:
    num_layers = 0
    runtime_arr_arr = []
    size_arr = []

    def load_model(self):
        raise NotImplementedError()

    def solve(self):
        model = gp.Model("prepartition")
        model.Params.TimeLimit = 3600

        # Create a matrix of decision variables of size (num_prepartitions, num_layers)
        self.partition_mat = np.array(
            [
                [model.addVar(vtype=GRB.BINARY) for j in range(self.num_layers)]
                for i in range(min(num_partitions, self.num_layers))
            ]
        )

        # Each layer only appear in one partition
        for j in range(self.partition_mat.shape[1]):
            model.addConstr(np.sum(self.partition_mat[:, j]) == 1)

        # Layers are partitioned continuously
        for i in range(self.partition_mat.shape[0] - 1):
            for j in range(self.partition_mat.shape[1] - 1):
                model.addConstr(
                    (self.partition_mat[i][j] == 1)
                    >> (
                        self.partition_mat[i][j + 1] + self.partition_mat[i + 1][j + 1]
                        == 1
                    )
                )
        for j in range(self.partition_mat.shape[1] - 1):
            model.addConstr(
                (self.partition_mat[-1][j] == 1) >> (self.partition_mat[-1][j + 1] == 1)
            )

        # No partitions are empty
        model.addConstr(self.partition_mat[0][0] == 1)
        model.addConstr(self.partition_mat[-1][-1] == 1)

        # Get layers that requires transmission
        need_trans_arr = [
            model.addVar(vtype=GRB.BINARY) for _ in range(self.num_layers - 1)
        ]
        for i in range(self.partition_mat.shape[0] - 1):
            for j in range(self.partition_mat.shape[1] - 1):
                need_trans_at_this_layer = model.addVar(vtype=GRB.BINARY)
                model.addConstr(
                    need_trans_at_this_layer
                    == gp.and_(
                        self.partition_mat[i][j], self.partition_mat[i + 1][j + 1]
                    )
                )
                model.addConstr(
                    (need_trans_at_this_layer == True) >> (need_trans_arr[j] == True)
                )

        # For each GPU, each partition's runtime is within [lb, ub] times of the average
        if self.num_layers > num_partitions:
            self.rt_norm_arr_arr = self.collect_normalized_runtimes_per_gpu_per_block()
            for rt_norm_arr in self.rt_norm_arr_arr:
                for rt_norm in rt_norm_arr:
                    lb_norm = self.lb / num_partitions
                    ub_norm = self.ub / num_partitions
                    model.addConstr(rt_norm >= lb_norm)
                    model.addConstr(rt_norm <= ub_norm)

        # Minimize transmission size
        trans_size = model.addVar(vtype=GRB.CONTINUOUS)
        model.addConstr(trans_size == np.dot(need_trans_arr, self.size_arr))
        model.setObjective(trans_size, GRB.MINIMIZE)

        start = time.time()
        model.optimize()
        end = time.time()
        print("Runtime: ", end - start)

        self.partition_mat_val = np.array(
            [
                [
                    get_gp_value(self.partition_mat[i][j])
                    for j in range(self.partition_mat.shape[1])
                ]
                for i in range(self.partition_mat.shape[0])
            ]
        )
        self.rt_norm_arr_arr_vals = np.array(
            [
                [get_gp_value(rt_norm) for rt_norm in rt_norm_arr]
                for rt_norm_arr in self.rt_norm_arr_arr
            ]
        )
        self.verify_partition_continuous()

    def collect_normalized_runtimes_per_gpu_per_block(self):
        rt_norm_arr_arr = []

        for gpu_idx in range(len(self.runtime_arr_arr)):
            rt_norm_arr = []
            runtime_arr = self.runtime_arr_arr[gpu_idx]

            for partition_idx in range(min(num_partitions, self.num_layers)):
                runtime_this_block = np.dot(
                    self.partition_mat[partition_idx], runtime_arr
                )
                rt_norm_arr.append(runtime_this_block / np.sum(runtime_arr))
            rt_norm_arr_arr.append(rt_norm_arr)
        return rt_norm_arr_arr

    def verify_partition_continuous(self):
        epsilon = 0.01
        assert np.all(np.diff(np.argwhere(self.partition_mat_val > epsilon)[:, 1]) == 1)

    def export_to_csv(self, savepath_mapping, savepath_size):
        """
        Export partitioning to a CSV with two columns (layer_name, prepartition_id), and
        export critical size to a CSV with one column.
        """
        raise NotImplementedError()


class DNNModelFmtv2(DNNModel):
    """
    Pre-partition using ONNX runtime, in folder `node-profile-no-const`.
    """

    def __init__(self, dnn_name, lb, ub, datadir):
        self.dnn_name = dnn_name
        self.lb = lb
        self.ub = ub
        self.datadir = datadir
        self.load_model()

    def load_model(self):
        self.layer_names, self.runtime_arr_arr = self._load_layer_names_and_runtimes()
        self.size_arr = self._load_sizes()

        self.num_layers = len(self.runtime_arr_arr[0])
        assert (
            len(self.size_arr) == self.num_layers - 1
        ), f"{len(self.size_arr)} != {self.num_layers - 1}"

    def export_to_csv(self, savepath_mapping, savepath_size):
        # Mapping
        df = []
        epsilon = 0.01
        block_part_id_arr = np.argwhere(self.partition_mat_val > epsilon)[:, 0]
        df = pd.DataFrame({"layer": self.layer_names, "block_id": block_part_id_arr})
        savepath_mapping.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(savepath_mapping, header=False, index=False)

        layers_need_trans = np.argwhere(block_part_id_arr[1:] - block_part_id_arr[:-1])[
            :, 0
        ]
        assert len(layers_need_trans) == min(num_partitions, self.num_layers) - 1
        size_need_trans = np.array(self.size_arr)[layers_need_trans]
        df = pd.DataFrame(size_need_trans, columns=["size_need_trans"])
        savepath_size.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(savepath_size, header=False, index=False)

    def _load_layer_names_and_runtimes(self):
        runtime_arr_arr = []
        for gpu_name in gpu_name_arr:
            df_runtime = pd.read_csv(
                self.datadir
                / "models"
                / "node-profile-no-const"
                / f"{self.dnn_name}-{gpu_name}.csv"
            )
            runtime_arr_arr.append(df_runtime.latency.tolist())
        layer_names = df_runtime.node
        return layer_names, runtime_arr_arr

    def _load_sizes(self):
        edge2sz = self._load_edge2sz()
        with open(
            self.datadir / "models" / "cuts-no-const" / f"{self.dnn_name}.json"
        ) as f:
            cuts = json.load(f)
        # Keep only intermediate cuts
        cuts = cuts[1:-2]

        size_arr = []
        for cut in cuts:
            sz = np.sum([edge2sz[e] for e in cut["cut"]])
            size_arr.append(sz)
        return size_arr

    def _load_edge2sz(self):
        with open(self.datadir / "models" / "shapes" / f"{self.dnn_name}.json") as f:
            shapes = json.load(f)
        ret = {}
        for edge in shapes:
            sz = self._compute_edge_sz(shapes[edge]["shape"], shapes[edge]["dtype"])
            ret[edge] = sz
        return ret

    @staticmethod
    def _compute_edge_sz(dims, dtype):
        count = 1
        for d in dims:
            # Take abs to make batch size dim positive
            count *= abs(d)

        if np.dtype(dtype) == np.dtype("float32"):
            itemsize = np.dtype("float16").itemsize
        elif np.dtype(dtype) == np.dtype("int64"):
            itemsize = np.dtype("int32").itemsize
        else:
            itemsize = np.dtype(dtype).itemsize

        return count * itemsize


def get_gp_value(v):
    if isinstance(v, gp.Var):
        return v.X
    elif isinstance(v, gp.LinExpr):
        return v.getValue()
    else:
        return v


def main(datadir):
    # Gradually relax requirement until solution found
    lb_ub_arr = [
        [0.7, 1.3],
        [0.5, 1.5],
        [0.5, 2.0],
        [0.5, 2.5],
        [0.5, 3.0],
        [0.5, 3.5],
        [0.5, 4.0],
        [0.4, 4.0],
    ]
    dnn_name_arr = pd.read_csv(datadir / "model_list.txt",
                               sep=" ", header=None)[0].to_numpy()

    for i, dnn_name in enumerate(dnn_name_arr):
        for lb, ub in lb_ub_arr:
            print("dnn_name, lb, ub:", dnn_name, lb, ub)
            try:
                m = DNNModelFmtv2(dnn_name, lb, ub, datadir)
                m.solve()
            except AttributeError as infeasible:
                continue

            tag = f"{num_partitions}-{lb}-{ub}"
            savepath_mapping = (
                Path("outputs")
                / 'prepartition_mappings'
                / dnn_name
                / f"contraction_mapping_ilpprepart{tag}.csv"
            )
            savepath_size = (
                Path("outputs")
                / 'prepartition_mappings'
                / dnn_name
                / f"contraction_trans-size_ilpprepart{tag}.csv"
            )
            m.export_to_csv(savepath_mapping, savepath_size)
            break


if __name__ == "__main__":
    main(datadir=Path('./data'))
