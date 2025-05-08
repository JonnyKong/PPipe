package edu.purdue.dsnl.clustersim;

import picocli.CommandLine;

import java.io.File;
import java.util.concurrent.Callable;
import java.util.stream.Stream;

@CommandLine.Command(name = "baseline")
public class SimpleSetup extends Main.ParentOptions implements Callable<Integer> {
    @CommandLine.Option(names = {"-w", "--worker"})
    public int workerCount = 40;

    @CommandLine.Parameters
    File modelFile;

    @Override
    public Integer call() throws Exception {
        var model = new SimpleModel(modelFile);
        var workers = Stream.generate(() -> (Worker) new SimpleWorker(model)).limit(workerCount).toList();
        var lb = new SimpleLoadBalancer(workers);
        var gen = new PoissonRequestGenerator(lb, load, duration, slo);
        Simulator.getInstance().doAllEvents();
        if (logFile != null) {
            lb.writeLog(logFile);
        }
        return 0;
    }
}
