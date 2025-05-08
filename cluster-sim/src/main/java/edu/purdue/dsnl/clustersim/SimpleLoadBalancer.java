package edu.purdue.dsnl.clustersim;

import java.util.Collections;
import java.util.Comparator;
import java.util.List;

public class SimpleLoadBalancer extends LoadBalancer {

    private final List<Worker> workers;

    // The amount of time needed to leave for future parts of the model
    private int slack = 0;

    public SimpleLoadBalancer(List<Worker> workers) {
        this.workers = workers;
    }

    public SimpleLoadBalancer(List<Worker> workers, int slack) {
        this.workers = workers;
        this.slack = slack;
    }

    @Override
    public void execute() {
        while (true) {
            var earliestWorker =
                    Collections.min(workers, Comparator.comparingInt(Worker::completionTime));
            int completionTime = Math.max(earliestWorker.completionTime(), time);
            Model model = earliestWorker.model;
            int bs = 0;
            while (!queue.isEmpty()) {
                bs = model.maxBatchSize(queue.get(0).deadline - slack - completionTime);
                if (bs <= 0) {
                    var r = queue.remove(0);
                    r.setDiscarded(true);
                    log.add(r);
                } else {
                    break;
                }
            }
            if (bs <= 0) {
                break;
            }

            if (queue.size() >= bs) {
                var br = queue.subList(0, bs);
                earliestWorker.acceptBatch(new Batch(br, earliestWorker.getWorkerId(), time));
                log.addAll(br);
                br.clear();
            } else {
                time = queue.get(0).deadline - slack - model.getTotalLatency(queue.size());
                Simulator.getInstance().addEvent(this);
                break;
            }
        }
    }
}
