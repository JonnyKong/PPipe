package edu.purdue.dsnl.clustersim;

import lombok.Getter;
import lombok.Setter;

public abstract class Worker extends Event {
    private static int workerIdCounter = 0;

    @Getter private int workerId = workerIdCounter++;

    /**
     * An optional slack field that stores how much time slack needs to be reserved after this
     * worker.
     */
    @Setter public int slack = 0;

    /** An optional field that gets populated by a scheduler. */
    @Setter public int plannedBs = 0;

    /**
     * An optional requests acceptor (load balancer or network transfer emulator) following this
     * worker.
     */
    @Setter public RequestsAcceptable nextRequestsAcceptor = null;

    public Model model;

    public int pipelineId;

    // An optioanl field used by the dispatcher
    @Getter public int reservedTill = 0;

    public abstract void acceptBatch(Batch batch);

    public abstract int completionTime();

    public <T extends RequestsAcceptable> T getNextRequestsAcceptor(Class<T> t) {
        return t.cast(nextRequestsAcceptor);
    }
}
