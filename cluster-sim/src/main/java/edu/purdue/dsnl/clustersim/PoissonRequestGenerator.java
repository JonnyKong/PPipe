package edu.purdue.dsnl.clustersim;

import java.util.random.RandomGenerator;

public class PoissonRequestGenerator extends RequestGenerator {
    private final RandomGenerator random = RandomGenerator.getDefault();

    public PoissonRequestGenerator(RequestsAcceptable acceptor, int load, int duration, int slo) {
        super(acceptor, load, duration, slo);
    }

    @Override
    protected int getNextInterarrival() {
        return (int) (1000000 * random.nextExponential() / load);
    }
}
