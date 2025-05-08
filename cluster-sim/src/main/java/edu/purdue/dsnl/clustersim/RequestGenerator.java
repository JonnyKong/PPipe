package edu.purdue.dsnl.clustersim;

import java.util.List;

public abstract class RequestGenerator extends Event {
    private final RequestsAcceptable acceptor;

    private final int duration;

    private final int slo;

    protected int count = 0;

    protected final int load;

    @Override
    public void execute() {
        acceptor.acceptRequests(List.of(new Request(count++, time, 0, time + slo)));
        time += getNextInterarrival();
        if (time < duration) {
            Simulator.getInstance().addEvent(this);
        }
    }

    public RequestGenerator(RequestsAcceptable acceptor, int load, int duration, int slo) {
        this.acceptor = acceptor;
        this.load = load;
        this.duration = duration;
        this.slo = slo;

        time = 0;
        Simulator.getInstance().addEvent(this);
    }

    protected abstract int getNextInterarrival();
}
