package edu.purdue.dsnl.clustersim;

import com.fasterxml.jackson.annotation.JsonIgnore;

import lombok.Data;

import java.util.LinkedList;
import java.util.List;

@Data
public class Batch {
    private static int batchIdCounter = 0;

    @JsonIgnore public final List<Request> requests;

    public final int batchId = batchIdCounter++;

    final int workerId;

    final int timeArriveWorker;

    int timeStartInference;

    int timeComplete;

    /** An optional field populated by ReservationLoadBalancer. */
    @JsonIgnore public LinkedList<Worker> reservedRoute;

    public Batch(List<Request> requests, int workerId, int timeArriveWorker) {
        this.requests = List.copyOf(requests);
        this.workerId = workerId;
        this.timeArriveWorker = timeArriveWorker;
        for (var r : requests) {
            r.setBatch(this);
        }
    }

    public int getBs() {
        return requests.size();
    }
}
