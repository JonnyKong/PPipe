/** Delays a batch by some time to simulate network transfers. */
package edu.purdue.dsnl.clustersim.ilp_partitioned;

import edu.purdue.dsnl.clustersim.*;

import lombok.RequiredArgsConstructor;

import java.util.LinkedList;
import java.util.List;
import java.util.Queue;

@RequiredArgsConstructor
public class RequestsDelayer extends Event implements RequestsAcceptable {

    private final int transSizeBytes;

    private final int bandwidth;

    private final RequestsAcceptable nextAcceptor;

    private final Queue<List<Request>> q = new LinkedList<>();

    public int reservedDl = 0;

    public int reservedUl = 0;

    public int networkLatency() {
        // int32 could overflow here
        return (int) Math.round((long) transSizeBytes * 8 / (bandwidth * 1000.0));
    }

    @Override
    public void acceptRequests(List<Request> requests) {
        for (Request r : requests) {
            r.timeArriveDelayer = Simulator.getInstance().getTime();
        }
        if (q.isEmpty()) {
            time = Simulator.getInstance().getTime() + requests.size() * networkLatency();
            Simulator.getInstance().addEvent(this);
        }
        q.add(requests);
    }

    /**
     * Only called when a batch is delayed for enough time and is ready to be sent to the next lb.
     */
    @Override
    public void execute() {
        var b = q.poll();
        if (nextAcceptor != null) {
            nextAcceptor.acceptRequests(b);
        }
        if (!q.isEmpty()) {
            time = Simulator.getInstance().getTime() + q.peek().size() * networkLatency();
            Simulator.getInstance().addEvent(this);
        }
    }
}
