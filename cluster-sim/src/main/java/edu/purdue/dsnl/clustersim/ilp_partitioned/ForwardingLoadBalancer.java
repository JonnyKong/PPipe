/**
 * A load balancer that simply forwards batches from previous workers to subsequent workers, without
 * doing any re-batching.
 */
package edu.purdue.dsnl.clustersim.ilp_partitioned;

import edu.purdue.dsnl.clustersim.*;

import java.util.Collections;
import java.util.Comparator;
import java.util.List;

public class ForwardingLoadBalancer extends LoadBalancer {

    private final List<Worker> workers;

    public ForwardingLoadBalancer(List<Worker> workers) {
        this.workers = workers;
    }

    @Override
    public void execute() {
        while (true) {
            if (queue.size() == 0) {
                break;
            }

            Worker earliestWorker =
                    Collections.min(workers, Comparator.comparingInt(Worker::completionTime));

            // Dequeue the requests that came from the same batch
            int bs = 1;
            while (bs < queue.size() && queue.get(bs).batch.batchId == queue.get(0).batch.batchId) {
                bs++;
            }

            var br = queue.subList(0, bs);
            earliestWorker.acceptBatch(new Batch(br, earliestWorker.getWorkerId(), time));
            log.addAll(br);
            br.clear();
        }
    }
}
