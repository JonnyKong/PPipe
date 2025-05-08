"""
    Contract DNN layers by dividing the runtime equally. Write contraction mapping to csv.
"""
from pathlib import Path

import pandas as pd

latency_root = Path('/export2/kong102/clusterserving_results/layer-timing')
mapping_root = Path('/home/kong102/ClusterServing/scripts/contraction_mappings')
models = ['resnet101', 'yolov5l', 'convnext_large', 'deeplabv3',  'efficientnet_b7', 'densedepth']
num_layer_groups = 10

# Use a reference GPU and BS to divide the layers into groups of equal runtimes
reference_gpu = 'A10-1'
reference_bs = 16


def read_layerwise_runtime(p):
    df = pd.read_csv(p, header=None)

    # Remove reformatting nodes
    df = df[~df.iloc[:, 0].str.startswith('Reformatting')]

    # Remove rows with runtime 0
    df = df.loc[~(df.iloc[:, 1:] == 0).all(axis=1)]

    return df


def split_by_equal_runtime():
    for m in models:
        df_mapping = read_layerwise_runtime(latency_root / m / reference_gpu / 'runtime.csv')
        df_mapping['cumsum'] = df_mapping.iloc[:, reference_bs].cumsum()
        df_mapping['layer_group_id'] = df_mapping['cumsum'] / \
            ((df_mapping['cumsum'].iloc[-1] + 1) / num_layer_groups)
        df_mapping['layer_group_id'] = df_mapping['layer_group_id'].astype(int)
        df_mapping = df_mapping[[0, 'layer_group_id']]

        df_mapping.to_csv(mapping_root / m / f'contraction_mapping_{num_layer_groups}.csv',
                          index=False, header=False)

        gpus = (latency_root / m).glob('*-[0-9]')
        for gpu in gpus:
            df = read_layerwise_runtime(gpu / 'runtime.csv')
            df = df.merge(df_mapping, on=[0], how='left')
            assert not df.layer_group_id.isnull().any()
            df = df.groupby(['layer_group_id']).sum()
            df.to_csv(gpu / f'runtime_{num_layer_groups}layers.csv', index=False, header=False)

            # # Also print runtime for manual partitioning for reference
            # for tag in ['ilpprepart10-0.7-1.3']:
            #     df_mapping_manual = pd.read_csv(mapping_root / m / f'contraction_mapping_{tag}.csv',
            #                                     header=None)
            #     df_mapping_manual = df_mapping_manual.rename(columns={1: 'layer_group_id'})
            #     df = read_layerwise_runtime(gpu / 'runtime.csv')
            #     df = df.merge(df_mapping_manual, on=[0], how='left')
            #     assert not df.layer_group_id.isnull().any()
            #     df = df.groupby(['layer_group_id']).sum()
            #     df.to_csv(gpu / f'runtime_{tag}.csv', index=False, header=False)


def print_merged_runtime_to_csv():
    # Print runtime for manual / ILP-prepartitioning for reference
    for m in models:
        gpus = (latency_root / m).glob('*-[0-9]')
        for gpu in gpus:
            df = read_layerwise_runtime(gpu / 'runtime.csv')

            for tag in ['manual', 'ilpprepart10-0.7-1.3']:
                df_mapping = pd.read_csv(mapping_root / m / f'contraction_mapping_{tag}.csv',
                                         header=None)
                df_mapping = df_mapping.rename(columns={1: 'layer_group_id'})

                df_merged = df.merge(df_mapping, on=[0], how='left')
                assert not df_merged.layer_group_id.isnull().any()
                df_merged = df_merged.groupby(['layer_group_id']).sum()
                df_merged.to_csv(gpu / f'runtime_{tag}.csv', index=False, header=False)


if __name__ == '__main__':
    # split_by_equal_runtime()
    print_merged_runtime_to_csv()
