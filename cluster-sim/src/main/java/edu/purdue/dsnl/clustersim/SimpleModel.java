package edu.purdue.dsnl.clustersim;

import lombok.Getter;
import tech.tablesaw.api.Table;
import tech.tablesaw.io.csv.CsvReadOptions;

import java.io.File;
import java.util.List;

public class SimpleModel implements Model {
    @Getter
    private final List<Integer> totalLatencies;

    SimpleModel(File modelFile) {
        totalLatencies = Table.read().csv(CsvReadOptions.builder(modelFile).header(false).build())
                .intColumn(0).asList();
    }
}
