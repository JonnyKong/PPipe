package edu.purdue.dsnl.clustersim.ilp_partitioned;

import edu.purdue.dsnl.clustersim.Batch;
import edu.purdue.dsnl.clustersim.LoadBalancer;
import edu.purdue.dsnl.clustersim.Simulator;
import edu.purdue.dsnl.clustersim.Worker;

import java.util.LinkedList;
import java.util.List;
import java.util.Optional;

public class ReservationFixedBsLoadBalancer extends LoadBalancer {
    private final List<Pipeline> pipelines;

    public ReservationFixedBsLoadBalancer(List<Pipeline> pipelines) {
        this.pipelines = pipelines;
    }

    @Override
    public void execute() {
        while (true) {
            // Look for the pipeline that can run a full batch the soonest
            Optional<ReservationOffer> r =
                    pipelines.stream()
                            .filter(p -> p.workerss.get(0).get(0).plannedBs <= queue.size())
                            .map(
                                    p ->
                                            ReservationOffer.reserve(
                                                    p, p.workerss.get(0).get(0).plannedBs))
                            .min(ReservationOffer.BY_QUEUING);

            if (!r.isPresent()) {
                // Cannot make up a batch
                break;
            }

            int queuingTime = r.get().queuingLatency;
            if (queuingTime == 0) {
                // No queuing, make reservation, start inference right away
                r.get().accept();

                Worker w = r.get().workers.get(0);
                var br = queue.subList(0, w.plannedBs);

                Batch b = new Batch(br, w.getWorkerId(), time);
                b.reservedRoute = new LinkedList<Worker>(r.get().workers);
                b.reservedRoute.pop();

                w.acceptBatch(b);
                log.addAll(br);
                br.clear();

            } else {
                // Need queuing. Delay reservation and inference until when no queuing will be
                // needed
                time += queuingTime;
                Simulator.getInstance().addEvent(this);
                break;
            }
        }
    }
}
