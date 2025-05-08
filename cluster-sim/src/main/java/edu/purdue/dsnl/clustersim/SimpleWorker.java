package edu.purdue.dsnl.clustersim;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;

public class SimpleWorker extends Worker {

    // Remember the last batch so as to send it to `nextLb`
    Batch lastBatch = null;

    protected final Queue<Batch> pendingBatches = new LinkedList<>();

    public SimpleWorker(Model model) {
        this.model = model;
    }

    @Override
    public void execute() {
        if (lastBatch != null && nextRequestsAcceptor != null) {
            List<Request> newRequests = new ArrayList<>();
            for (Request r : lastBatch.requests) {
                // Make a copy of the request so that next lb mutating the request will not affect
                // the previous lb
                Request r_ = new Request(r.requestId, r.arrivalTime, r.streamId, r.deadline);
                r_.batch = lastBatch;
                newRequests.add(r_);
            }
            nextRequestsAcceptor.acceptRequests(newRequests);
            lastBatch = null;
        }

        var b = pendingBatches.poll();
        if (b != null) {
            b.setTimeStartInference(time);
            time += model.getTotalLatency(b.getBs());
            b.setTimeComplete(time);
            Simulator.getInstance().addEvent(this);

            lastBatch = b;
        }
    }

    @Override
    public void acceptBatch(Batch batch) {
        pendingBatches.add(batch);
        if (!Simulator.getInstance().containsEvent(this)) {
            time = Simulator.getInstance().getTime();
            execute();
        }
    }

    @Override
    public int completionTime() {
        return time + pendingBatches.stream().mapToInt(b -> model.getTotalLatency(b.getBs())).sum();
    }
}
