package edu.purdue.dsnl.clustersim.s3dnn;

import edu.purdue.dsnl.clustersim.Batch;
import edu.purdue.dsnl.clustersim.Simulator;
import edu.purdue.dsnl.clustersim.Request;
import edu.purdue.dsnl.clustersim.Worker;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

public class S3dnnWorker extends Worker {
    private static final double THOLD_UTIL = 0.8;

    private final S3dnnModel model;

    private final List<S3dnnBatch> batches = new ArrayList<>();

    private class S3dnnBatch {
        final Batch batch;

        int deadline;

        int nextLayer;

        S3dnnBatch(Batch batch) {
            this.batch = batch;
            deadline = Collections.min(batch.getRequests(), Comparator.comparingInt(Request::getDeadline))
                    .getDeadline();
        }

        int getBs() {
            return batch.getBs();
        }

        int getSlack() {
            return deadline - time - model.getRemainingTime(getBs(), nextLayer);
        }
    }

    S3dnnWorker(S3dnnModel model) {
        this.model = model;
    }

    @Override
    public void execute() {
        if (batches.isEmpty()) {
            return;
        }

        var q = new ArrayList<>(batches);
        q.sort(Comparator.comparingInt(S3dnnBatch::getSlack));
        var h = q.remove(0);

        var g = new ArrayList<S3dnnBatch>();
        g.add(h);

        double hUtil = model.getUtilization(h.getBs(), h.nextLayer);
        if (hUtil < THOLD_UTIL) {
            double util = hUtil;
            for (var t : q) {
                double u = model.getUtilization(t.getBs(), t.nextLayer);
                if (util + u <= 1) {
                    g.add(t);
                    util += u;
                }
            }

            if (util < THOLD_UTIL) {
                var hpo = q.stream().dropWhile(g::contains).findFirst();
                if (hpo.isPresent()) {
                    var hp = hpo.get();
                    int hSlack = h.getSlack();
                    for (int i = 1; hp.nextLayer + i < model.getLayerCount(); i++) {
                        hSlack -= model.getLatency(hp.getBs(), hp.nextLayer + i);
                        if (hSlack < 0) {
                            break;
                        }
                        if (util + model.getUtilization(hp.getBs(), hp.nextLayer + i) <= 1) {
                            g.clear();
                            g.add(hp);
                        }
                    }
                }
            }
        }

        time += g.stream().mapToDouble(b -> {
            if (b.nextLayer == 0) {
                b.batch.setTimeStartInference(time);
            }
            b.nextLayer++;
            return model.getLatency(b.getBs(), b.nextLayer - 1);
        }).max().getAsDouble();
        batches.removeIf(b -> {
            boolean complete = b.nextLayer >= model.getLayerCount();
            if (complete) {
                b.batch.setTimeComplete(time);
            }
            return complete;
        });
        Simulator.getInstance().addEvent(this);
    }

    @Override
    public void acceptBatch(Batch batch) {
        batches.add(new S3dnnBatch(batch));
        if (!Simulator.getInstance().containsEvent(this)) {
            time = Simulator.getInstance().getTime();
            execute();
        }
    }

    @Override
    public int completionTime() {
        return time + batches.stream().mapToInt(b -> model.getWeightedRemainingTime(b.getBs(), b.nextLayer)).sum();
    }
}
