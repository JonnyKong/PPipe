/** A load balancer that always uses the BS specified by the earliest worker. */
package edu.purdue.dsnl.clustersim.ilp_partitioned;

import edu.purdue.dsnl.clustersim.*;

import java.util.Collections;
import java.util.Comparator;
import java.util.List;

public class PlannedBsLoadBalancer extends LoadBalancer {

    private final List<Worker> workers;

    public PlannedBsLoadBalancer(List<Worker> workers) {
        this.workers = workers;
    }

    @Override
    public void execute() {
        while (true) {
            Worker earliestWorker =
                    Collections.min(workers, Comparator.comparingInt(Worker::completionTime));
            int completionTime = Math.max(earliestWorker.completionTime(), time);
            int slack = earliestWorker.slack;
            Model model = earliestWorker.model;
            int bs = earliestWorker.plannedBs;

            if (queue.size() >= bs) {
                var br = queue.subList(0, bs);
                earliestWorker.acceptBatch(new Batch(br, earliestWorker.getWorkerId(), time));
                log.addAll(br);
                br.clear();
            } else {
                break;
            }
        }
    }
}
