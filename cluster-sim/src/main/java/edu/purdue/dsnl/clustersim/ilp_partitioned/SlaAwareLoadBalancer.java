/** A load balancer that reads the slack from the next worker to set the batch size. */
package edu.purdue.dsnl.clustersim.ilp_partitioned;

import edu.purdue.dsnl.clustersim.*;

import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.stream.IntStream;

public class SlaAwareLoadBalancer extends LoadBalancer {

    private final List<Worker> workers;

    private int examinedHead = -1;

    public SlaAwareLoadBalancer(List<Worker> workers) {
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
            var totalLatencies = model.getTotalLatencies();
            var delayer = earliestWorker.getNextRequestsAcceptor(RequestsDelayer.class);
            int networkLatency = delayer != null ? delayer.networkLatency() : 0;
            int bs = 0;
            int minBs = (int) Math.round(0.95 * earliestWorker.plannedBs);

            while (!queue.isEmpty()) {
                // Take the largest possible bs subject to SLA, up to the planned bs
                int remaining = queue.get(0).deadline - slack - completionTime;
                bs = IntStream.range(0, earliestWorker.plannedBs)
                        .filter(i -> totalLatencies.get(i) + (i + 1) * networkLatency > remaining)
                        .findFirst()
                        .orElse(earliestWorker.plannedBs);

                if (queue.get(0).requestId != examinedHead && bs < minBs || bs <= 0) {
                    var r = queue.remove(0);
                    r.setDiscarded(true);
                    log.add(r);
                } else {
                    break;
                }
            }
            if (queue.isEmpty()) {
                break;
            }

            if (queue.size() >= bs) {
                var br = queue.subList(0, bs);
                earliestWorker.acceptBatch(new Batch(br, earliestWorker.getWorkerId(), time));
                log.addAll(br);
                br.clear();
            } else {
                examinedHead = queue.get(0).requestId;
                time = queue.get(0).deadline - slack - model.getTotalLatency(queue.size() + 1) + 1;
                Simulator.getInstance().addEvent(this);
                break;
            }
        }
    }
}
