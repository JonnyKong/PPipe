package edu.purdue.dsnl.clustersim;

import java.util.Arrays;

import tech.tablesaw.api.DoubleColumn;
import tech.tablesaw.api.Table;
import tech.tablesaw.io.csv.CsvReadOptions;

public class TraceRequestGenerator extends RequestGenerator {
    private double[] interarrivalTimes;

    public TraceRequestGenerator(RequestsAcceptable acceptor, int load, int duration, int slo,
            String tracePath) {
        super(acceptor, load, duration, slo);

        var arrivalTimes = loadArrivalTimesFromTrace(tracePath);
        var scaledArrivalTimes = scaleTrace(arrivalTimes, load);
        interarrivalTimes = diff(scaledArrivalTimes);
    }

    @Override
    protected int getNextInterarrival() {
        return (int) (interarrivalTimes[count % interarrivalTimes.length] * 1e6);
    }

    private double[] loadArrivalTimesFromTrace(String tracePath) {
        Table trace = Table.read().csv(CsvReadOptions.builder(tracePath).build());

        DoubleColumn endTimestamps = (DoubleColumn) trace.column("end_timestamp");
        DoubleColumn durations = (DoubleColumn) trace.column("duration");

        DoubleColumn startTimestamps = endTimestamps.subtract(durations);
        return startTimestamps.asDoubleArray();
    }

    private static double[] scaleTrace(double[] arrivalTimes, float load) {
        Arrays.sort(arrivalTimes);
        var durationTotal = arrivalTimes[arrivalTimes.length - 1] - arrivalTimes[0];
        var currentQps = (arrivalTimes.length - 1) / durationTotal;
        var scaleUpFactor = load / currentQps;

        var durationPerSegment = durationTotal / scaleUpFactor;
        return Arrays.stream(arrivalTimes)
                .map(x -> x % durationPerSegment)
                .sorted()
                .toArray();
    }

    private static double[] diff(double[] array) {
        if (array.length < 2) {
            throw new IllegalArgumentException("Array must have at least two elements");
        }

        double[] diff = new double[array.length - 1];
        for (int i = 0; i < array.length - 1; i++) {
            diff[i] = array[i + 1] - array[i];
        }
        return diff;
    }
}
