package edu.purdue.dsnl.clustersim.ilp_partitioned;

import edu.purdue.dsnl.clustersim.*;

import it.unimi.dsi.fastutil.ints.IntRBTreeSet;

import java.util.Comparator;
import java.util.LinkedList;
import java.util.List;

public class ReservationSlaLoadBalancer extends LoadBalancer {

    private final List<Pipeline> pipelines;

    private final int sla;

    private final IntRBTreeSet retriedReqIds = new IntRBTreeSet();

    public ReservationSlaLoadBalancer(List<Pipeline> pipelines, int sla) {
        this.pipelines = pipelines;
        this.sla = sla;
    }

    @Override
    public void execute() {
        while (true) {
            int deadline = queue.get(0).deadline;
            Pipeline pSelected =
                    pipelines.stream()
                            .map(
                                    p ->
                                            ReservationOffer.reserve(
                                                    p, p.workerss.get(0).get(0).plannedBs))
                            // no need to select bs=1 pipelines if SLA violated
                            .min(Comparator.comparingInt(
                                    o -> o.queuingLatency + ((o.bs == 1 && o.completionTime() > deadline) ? sla : 0)))
                            .get()
                            .pipeline;
            int plannedBs = pSelected.workerss.get(0).get(0).plannedBs;

            if (retriedReqIds.contains(queue.get(0).requestId)) {
                // Find the max BS without exceeding SLA
                ReservationOffer r = null;
                for (int bs = plannedBs; bs > 0; bs--) {
                    ReservationOffer r_ = ReservationOffer.reserve(pSelected, bs);
                    if (r_.completionTime() <= deadline) {
                        r = r_;
                        break;
                    }
                }
                if (r == null) {
                    // First request queued for too long, drop it
                    Request req = queue.remove(0);
                    req.pipelineId = pSelected.id;
                    req.setDiscarded(true);
                    req.dropCause = 1;
                    log.add(req);
                    if (queue.isEmpty()) {
                        break;
                    }
                    continue;
                }

                if (queue.size() >= r.bs) {
                    r.accept();
                    Worker w = r.workers.get(0);

                    var br = queue.subList(0, r.bs);
                    for (Request req : br) {
                        req.pipelineId = pSelected.id;
                    }
                    Batch b = new Batch(br, w.getWorkerId(), time);
                    b.reservedRoute = new LinkedList<Worker>(r.workers);
                    b.reservedRoute.pop();

                    w.acceptBatch(b);
                    log.addAll(br);
                    retriedReqIds.headSet(br.get(br.size() - 1).requestId + 1).clear();
                    br.clear();

                    if (queue.isEmpty()) {
                        break;
                    }

                } else {
                    // Wait to get more requests, up till bs+1 will violate
                    ReservationOffer rSmall = ReservationOffer.reserve(pSelected, queue.size() + 1);

                    if (rSmall.completionTime() > deadline) {
                        // Corner case of runtime not monotonic with bs
                        Request req = queue.remove(0);
                        req.pipelineId = pSelected.id;
                        req.setDiscarded(true);
                        req.dropCause = 2;
                        log.add(req);
                        if (queue.isEmpty()) {
                            break;
                        }
                        continue;
                    }

                    int waitFor = deadline - rSmall.completionTime() + rSmall.queuingLatency + 1;
                    assert waitFor > 0;
                    time += waitFor;
                    Simulator.getInstance().addEvent(this);
                    break;
                }

            } else {
                int maxBs = plannedBs;
                // This is empirically determined
                int minBs = (int) Math.round(plannedBs * 0.95);

                // Look for the largest BS satisfying SLA
                ReservationOffer r = null;
                ReservationOffer rMax = null;
                for (int bs = maxBs; bs >= minBs; bs--) {
                    ReservationOffer r_ = ReservationOffer.reserve(pSelected, bs);
                    if (bs == maxBs) {
                        rMax = r_;
                    }
                    if (r_.completionTime() <= deadline) {
                        r = r_;
                        break;
                    }
                }

                if (r == null) {
                    // First request queued for too long, drop it
                    Request req = queue.remove(0);
                    req.pipelineId = pSelected.id;
                    req.setDiscarded(true);
                    req.dropCause = rMax.queuingLatency;
                    log.add(req);
                    if (queue.isEmpty()) {
                        break;
                    }
                } else {
                    retriedReqIds.add(queue.get(0).requestId);
                }
            }
        }
    }

    /**
     * Select the pipeline with the minimal efficiency loss under the max BS size that can finish
     * within (SLA-queuing).
     */
    private Pipeline selectPipeline() {
        float maxEfficiency = 0f;
        Pipeline pSelected = null;

        for (Pipeline p : pipelines) {
            int deadline = time + sla;
            int plannedBs = p.workerss.get(0).get(0).plannedBs;

            // Find the max BS under SLA - queuing_time
            float xputFull = p.getXput(plannedBs);
            float xputPartial = 0f;
            int bs = plannedBs;
            for (; bs > 0; bs--) {
                xputPartial = p.getXput(bs);
                ReservationOffer rPartial = ReservationOffer.reserve(p, bs);
                if (rPartial.completionTime() <= deadline) {
                    break;
                }
            }

            float efficiency = xputPartial / xputFull;
            if (efficiency > maxEfficiency) {
                maxEfficiency = efficiency;
                pSelected = p;
            }
        }

        return pSelected;
    }
}
