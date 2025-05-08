package edu.purdue.dsnl.clustersim.ilp_partitioned;

import edu.purdue.dsnl.clustersim.*;

import java.util.LinkedList;
import java.util.List;

public class ReservationForwardingLoadBalancer extends LoadBalancer {

    @Override
    public void execute() {
        // Dequeue the requests that came from the same batch
        int bs = 1;
        while (bs < queue.size() && queue.get(bs).batch.batchId == queue.get(0).batch.batchId) {
            bs++;
        }
        var br = queue.subList(0, bs);

        LinkedList<Worker> route = br.get(0).batch.reservedRoute;

        Worker w = route.pop();

        Batch b = new Batch(br, w.getWorkerId(), time);
        b.reservedRoute = route;
        w.acceptBatch(b);
        log.addAll(br);
        br.clear();
    }
}
