package edu.purdue.dsnl.clustersim.manual_partitioned;

import edu.purdue.dsnl.clustersim.*;

import picocli.CommandLine;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.stream.Stream;

@CommandLine.Command(name = "manual_partitioned")
public class ManualPartitionedSetup extends Main.ParentOptions implements Callable<Integer> {
    @CommandLine.Option(names = {"--logdir"})
    public File logDir;

    @CommandLine.Parameters(paramLabel = "latfiles")
    public File[] latFiles;

    int[] slos = {5367, 17575, 7058};
    int[] numWorkers = {6, 7, 10};
    int[] mpsMultipliers = new int[] {1, 4, 1};

    @Override
    public Integer call() throws Exception {
        // Calculate backwards the slacks needed at each lb
        int[] slacks = new int[slos.length];
        slacks[slacks.length - 1] = 0;
        for (int i = slacks.length - 2; i >= 0; i--) {
            slacks[i] = slacks[i + 1] + slos[i + 1];
        }

        List<ManualPartitionedModel> models = ManualPartitionedModel.getPartitionedModels(latFiles);

        List<SimpleLoadBalancer> lbs = new ArrayList<>();
        List<List<SimpleWorker>> workerss = new ArrayList<>();

        // Instantiate workers and load balancers
        for (int i = 0; i < models.size(); i++) {
            ManualPartitionedModel model = models.get(i);
            var workers =
                    Stream.generate(() -> new SimpleWorker(model))
                            .limit(numWorkers[i] * mpsMultipliers[i])
                            .toList();
            var lb = new SimpleLoadBalancer(List.copyOf(workers), slacks[i]);

            // Maintain reference to avoid being GC-ed
            workerss.add(workers);
            lbs.add(lb);
        }

        // Link workers to next stage load balancers
        for (int i = 0; i < models.size() - 1; i++) {
            SimpleLoadBalancer nextLb = lbs.get(i + 1);
            workerss.get(i).forEach(w -> w.setNextRequestsAcceptor(nextLb));
        }

        var gen = new PoissonRequestGenerator(lbs.get(0), load, duration, slo);

        Simulator.getInstance().doAllEvents();
        if (logDir != null) {
            int lbIdx = 0;
            for (SimpleLoadBalancer lb : lbs) {
                lb.writeLog(new File(logDir, String.format("manual_partitioned_%d.csv", lbIdx++)));
            }
        }

        return 0;
    }
}
