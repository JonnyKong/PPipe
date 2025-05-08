package edu.purdue.dsnl.clustersim.ilp_partitioned;

import edu.purdue.dsnl.clustersim.*;

import it.unimi.dsi.fastutil.ints.IntArrayList;

import lombok.RequiredArgsConstructor;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

/**
 * Represents a pending reservation for each partition in a pipeline. Calling the apply() method
 * will consolidate the reservation.
 */
@RequiredArgsConstructor
class ReservationOffer {
    public final int bs;

    public final int queuingLatency;

    public final Pipeline pipeline;

    public final List<Worker> workers;

    public final IntArrayList reservedTills;

    public final IntArrayList reservedNets;

    public void accept() {
        for (int i = 0; i < workers.size(); i++) {
            workers.get(i).reservedTill = reservedTills.getInt(i);
        }
        for (int i = 0; i < workers.size() - 1; i++) {
            workers.get(i).getNextRequestsAcceptor(RequestsDelayer.class).reservedUl = reservedNets.getInt(i);
            workers.get(i + 1).getNextRequestsAcceptor(RequestsDelayer.class).reservedDl = reservedNets.getInt(i);
        }
    }

    public int completionTime() {
        return reservedTills.getInt(reservedTills.size() - 1);
    }

    /** Given a pipeline, simulate the inference of a batch of a given size. */
    public static ReservationOffer reserve(Pipeline p, int bs) {
        List<Worker> reservedWorkers = new ArrayList<>();
        IntArrayList reservedTills = new IntArrayList();
        IntArrayList reservedNets = new IntArrayList();
        int workLatency = 0;
        int now = Simulator.getInstance().getTime();
        int tGlobal = now;

        // Simulate batch inference
        for (int i = 0; i < p.workerss.size(); i++) {
            int netLatency = 0;
            if (i != 0) {
                netLatency = bs * p.workerss.get(i - 1).get(0)
                        .getNextRequestsAcceptor(RequestsDelayer.class).networkLatency();
            }
            int infLatency = p.models.get(i).getTotalLatency(bs);
            workLatency += netLatency + infLatency;

            int tBest = Integer.MAX_VALUE;
            int tNetBest = 0;
            int tInfBest = 0;
            Worker wBest = null;
            for (var w : p.workerss.get(i)) {
                int t = tGlobal;

                int tNet = 0;
                if (i != 0) {
                    int tUl = reservedWorkers.get(i - 1).getNextRequestsAcceptor(RequestsDelayer.class).reservedUl;
                    int tDl = w.getNextRequestsAcceptor(RequestsDelayer.class).reservedDl;
                    t = tNet = Math.max(Math.max(tUl, tDl), t) + netLatency;
                }

                int tInf = t = Math.max(w.getReservedTill(), t) + infLatency;

                if (i != p.workerss.size() - 1) {
                    t = Math.max(w.getNextRequestsAcceptor(RequestsDelayer.class).reservedUl, t);
                }

                if (t < tBest) {
                    tBest = t;
                    tNetBest = tNet;
                    tInfBest = tInf;
                    wBest = w;
                }
            }
            if (i != 0) {
                reservedNets.add(tNetBest);
            }
            reservedTills.add(tInfBest);
            reservedWorkers.add(wBest);
            tGlobal = tInfBest;
        }

        int queuingLatency = tGlobal - now - workLatency;
        return new ReservationOffer(bs, queuingLatency, p, reservedWorkers, reservedTills, reservedNets);
    }

    public static final Comparator<ReservationOffer> BY_QUEUING = Comparator.comparingInt(r -> r.queuingLatency);
}
