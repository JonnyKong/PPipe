package edu.purdue.dsnl.clustersim.manual_partitioned;

import tech.tablesaw.api.Table;
import edu.purdue.dsnl.clustersim.Model;
import lombok.Getter;
import tech.tablesaw.io.csv.CsvReadOptions;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

public class ManualPartitionedModel implements Model {
    @Getter
    private final List<Integer> totalLatencies;

    private ManualPartitionedModel(List<Integer> totalLatencies) {
        this.totalLatencies = totalLatencies;
    }

    /**
     * Given the per-layer runtimes of a DNN, a list of cutpoints to partition the model,
     * return a list of UnsyncedModel instances.
     *
     * @param latFiles          a list of latency per-layer runtime CSVs
     * @param partitionPoints   1-indexed layer ID, denotes the end of each partition, inclusive
     */
    public static List<ManualPartitionedModel> getPartitionedModels(File latFile, int[] partitionPoints) {
        var latTable = Table.read().csv(CsvReadOptions.builder(latFile).header(false).build());
        int layerStart = 1;
        ArrayList<ManualPartitionedModel> ret = new ArrayList<>();

        // For each model partition
        for (int layerEnd : partitionPoints) {
            // Sum up latency of layers in this partition for each bs
            ArrayList<Integer> totalLatencies = new ArrayList<>();
            for (int j = 1; j < latTable.columnCount(); j++) {
                int lat = 0;
                for (int i = layerStart; i < layerEnd + 1; i++) {
                    lat += latTable.row(i - 1).getInt(j);
                }
                totalLatencies.add(lat);
            }

            // Ensure non-decreasing, otherwise infinite loop
            for (int j = 0; j < totalLatencies.size() - 1; j++) {
                if (totalLatencies.get(j + 1) <= totalLatencies.get(j)) {
                    totalLatencies.set(j + 1, totalLatencies.get(j) + 1);
                }
            }

            ret.add(new ManualPartitionedModel(totalLatencies));
            layerStart = layerEnd + 1;
        }
        return ret;
    }


    /**
     * Given the runtime table for each DNN partition, return a list of corresponding
     * UnsyncedModel instances.
     *
     * @param latFiles          a list of latency CSVs
     */
    public static List<ManualPartitionedModel> getPartitionedModels(File[] latFiles) {
        List<ManualPartitionedModel> ret = new ArrayList<>();
        for (File f : latFiles) {
            List<Integer> l = Table.read().csv(CsvReadOptions.builder(f).header(false).build())
                        .intColumn(0).asList();
            for (int i = 1; i < l.size(); i++) {
                if (l.get(i) <= l.get(i - 1)) {
                    l.set(i, l.get(i - 1) + 1);
                }
            }
            ret.add(new ManualPartitionedModel(l));
        }
        return ret;
    }
}
