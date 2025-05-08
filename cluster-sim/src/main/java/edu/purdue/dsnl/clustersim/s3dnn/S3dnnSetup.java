package edu.purdue.dsnl.clustersim.s3dnn;


import edu.purdue.dsnl.clustersim.*;
import picocli.CommandLine;

import java.io.File;
import java.io.IOException;
import java.util.concurrent.Callable;
import java.util.stream.Stream;

@CommandLine.Command(name = "s3dnn")
public class S3dnnSetup extends Main.ParentOptions implements Callable<Integer> {
    @CommandLine.Option(names = {"-w", "--worker"})
    public int workerCount = 40;

    @CommandLine.Parameters
    File modelLat;

    @CommandLine.Parameters
    File modelUtil;

    @Override
    public Integer call() throws IOException {
        var model = new S3dnnModel(modelLat, modelUtil);
        var workers = Stream.generate(() -> (Worker) new S3dnnWorker(model)).limit(workerCount).toList();
        var lb = new SimpleLoadBalancer(workers);
        var gen = new PoissonRequestGenerator(lb, load, duration, slo);
        Simulator.getInstance().doAllEvents();
        if (logFile != null) {
            lb.writeLog(logFile);
        }
        return 0;
    }
}
