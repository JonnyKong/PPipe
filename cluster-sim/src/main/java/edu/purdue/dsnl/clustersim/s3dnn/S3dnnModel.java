package edu.purdue.dsnl.clustersim.s3dnn;

import edu.purdue.dsnl.clustersim.Model;
import lombok.Data;
import lombok.Getter;
import tech.tablesaw.api.Table;
import tech.tablesaw.io.csv.CsvReadOptions;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

public class S3dnnModel implements Model {
    private final List<List<Layer>> batches = new ArrayList<>();

    @Getter
    private final List<Integer> totalLatencies = new ArrayList<>();

    @Data
    static class Layer {
        final int latency;
        final double utilization;
        int remainingTime;
        int weightedRemainingTime;
    }

    S3dnnModel(File latFile, File utilFile) {
        var lat = Table.read().csv(CsvReadOptions.builder(latFile).header(false).build());
        var util = Table.read().csv(CsvReadOptions.builder(utilFile).header(false).build());
        for (int i = 1; i < lat.columnCount(); ++i) {
            var b = Table.create(lat.column(i).setName("lat" + i), util.column(i).setName("util" + i)).stream()
                    .map(r -> new Layer(r.getInt(0), r.getDouble(1) / 100.0)).toList();

            int cum = 0;
            int weightedCum = 0;
            for (int j = b.size() - 1; j >= 0; --j) {
                var l = b.get(j);
                cum += l.latency;
                l.remainingTime = cum;
                weightedCum += (int) (l.latency * l.utilization);
                l.weightedRemainingTime = weightedCum;
            }

            batches.add(b);
            totalLatencies.add((int) lat.intColumn(i).sum());
        }
    }

    int getLatency(int bs, int layer) {
        return batches.get(bs - 1).get(layer).latency;
    }

    double getUtilization(int bs, int layer) {
        return batches.get(bs - 1).get(layer).utilization;
    }

    int getRemainingTime(int bs, int layer) {
        return batches.get(bs - 1).get(layer).remainingTime;
    }

    int getWeightedRemainingTime(int bs, int layer) {
        return batches.get(bs - 1).get(layer).weightedRemainingTime;
    }

    int getLayerCount() {
        return batches.get(0).size();
    }
}
