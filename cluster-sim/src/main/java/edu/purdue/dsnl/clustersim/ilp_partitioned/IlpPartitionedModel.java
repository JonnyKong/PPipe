package edu.purdue.dsnl.clustersim.ilp_partitioned;

import edu.purdue.dsnl.clustersim.*;

import lombok.Getter;

import tech.tablesaw.api.Table;
import tech.tablesaw.io.csv.CsvReadOptions;
import tech.tablesaw.selection.Selection;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

public class IlpPartitionedModel implements Model {
    @Getter private final List<Integer> totalLatencies = new ArrayList<>();

    public final int transSize;

    public IlpPartitionedModel(
            File latencyFile,
            File transSizeFile,
            int startLayerId,
            int endLayerId) {

        // Read latency table
        Table latencyTable =
                Table.read().csv(CsvReadOptions.builder(latencyFile).header(false).build());
        var range = Selection.withRange(
                startLayerId,
                Math.min(endLayerId, latencyTable.rowCount()));
        for (int i = 0; i < latencyTable.columnCount(); i++) {
            totalLatencies.add((int) latencyTable.intColumn(i).where(range).sum());
        }

        if (transSizeFile.exists()) {
            Table transSizeTable =
                    Table.read().csv(CsvReadOptions.builder(transSizeFile).header(false).build());
            this.transSize =
                    (endLayerId - 1 < transSizeTable.intColumn(0).size())
                            ? transSizeTable.intColumn(0).get(endLayerId - 1)
                            : 0;
        } else {
            this.transSize = 0;
        }
    }
}
