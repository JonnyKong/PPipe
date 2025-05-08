package edu.purdue.dsnl.clustersim;

import com.fasterxml.jackson.annotation.JsonUnwrapped;
import lombok.Data;

@Data
public class Request {
    public final int requestId;

    public final int arrivalTime;

    public final int streamId;

    public final int deadline;

    public boolean discarded;

    public int timeArriveDelayer;

    public int timeArriveLb;
    
    public int pipelineId = -1;

    public int queueLen;

    public int dropCause = -1;

    @JsonUnwrapped
    public Batch batch;
}
