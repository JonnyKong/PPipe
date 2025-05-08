package edu.purdue.dsnl.clustersim;

import java.util.List;
import java.util.stream.IntStream;

public interface Model {
    List<Integer> getTotalLatencies();

    default int getTotalLatency(int bs) {
        return getTotalLatencies().get(bs - 1);
    }

    default int maxBatchSize(int slack) {
        var totalLatencies = getTotalLatencies();
        return IntStream.range(0, totalLatencies.size())
                .filter(i -> totalLatencies.get(i) > slack).findFirst().orElse(totalLatencies.size());
    }
}
