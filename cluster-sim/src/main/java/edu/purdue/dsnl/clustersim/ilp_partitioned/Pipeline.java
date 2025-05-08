package edu.purdue.dsnl.clustersim.ilp_partitioned;

import edu.purdue.dsnl.clustersim.LoadBalancer;
import edu.purdue.dsnl.clustersim.SimpleWorker;
import edu.purdue.dsnl.clustersim.Worker;
import edu.purdue.dsnl.clustersim.ilp_partitioned.IlpPartitionedSetup.LbChoice;
import it.unimi.dsi.fastutil.doubles.DoubleArrayList;
import lombok.Cleanup;
import lombok.SneakyThrows;
import org.json.JSONObject;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Stream;

public class Pipeline {
    private static int idCounter = 0;

    public final int id = idCounter++;

    public List<List<Worker>> workerss;

    public List<LoadBalancer> lbs = new ArrayList<>();

    public List<IlpPartitionedModel> models = new ArrayList<>();

    private final File latencyRoot;

    private final File mappingRoot;

    public Pipeline(JSONObject cfg, File latencyRoot, File mappingRoot, int bandwidth, LbChoice lbChoice) {
        this.latencyRoot = latencyRoot;
        this.mappingRoot = mappingRoot;

        // Calculate backwards the slacks needed at each lb
        var partitions = cfg.getJSONArray("partitions");
        int[] slacks = new int[partitions.length()];
        slacks[slacks.length - 1] = 0;
        for (int i = slacks.length - 2; i >= 0; i--) {
            int latInfer = partitions.getJSONObject(i + 1).getInt("lat_infer");
            int latTrans = partitions.getJSONObject(i + 1).getInt("lat_trans");
            int rt = latInfer + latTrans;
            slacks[i] = slacks[i + 1] + rt;
        }

        workerss = Stream.generate(ArrayList<Worker>::new).limit(partitions.length())
                .map(w -> (List<Worker>) w).toList();
        for (int part_id = 0; part_id < partitions.length(); part_id++) {
            var partition = partitions.getJSONObject(part_id);
            IlpPartitionedModel model = createModelFromJson(partition);
            models.add(model);

            int bs = partition.getInt("bs");
            int mps = partition.getInt("mps") + 1;
            int numWorkers = (int) Math.round(partition.getDouble("num_gpu") * mps);
            int gpuPerServer = 0;
            LoadBalancer lb = null;
            if (part_id != partitions.length() - 1) {
                if (lbChoice == LbChoice.SLA_AWARE) {
                    lb = new SlaAwareLoadBalancer(workerss.get(part_id + 1));
                } else if (lbChoice == LbChoice.RESERVATION) {
                    lb = new ReservationForwardingLoadBalancer();
                }
                lbs.add(lb);
                gpuPerServer = Math.max(partition.getInt("num_gpu_per_server"),
                        partitions.getJSONObject(part_id + 1).getInt("num_gpu_per_server"));
            }
            RequestsDelayer d = null;
            for (int i = 0; i < numWorkers; i++) {
                // TODO handle ul and dl using separate delayers
                if (gpuPerServer == 0 || i % (gpuPerServer * mps) == 0) {
                    d = new RequestsDelayer(model.transSize, bandwidth, lb);
                }
                Worker w = new SimpleWorker(model);
                w.setSlack(slacks[part_id]);
                w.setPlannedBs(bs);
                w.pipelineId = id;
                w.setNextRequestsAcceptor(d);
                workerss.get(part_id).add(w);
            }
        }
    }

    /** Given a DNN partition JSON object, return the corresponding IlpPartitionedModel. */
    @SneakyThrows(IOException.class)
    private IlpPartitionedModel createModelFromJson(JSONObject partition) {
        File latencyFile =
                new File(
                        String.format(
                                "%s/%s-%s-%d.csv",
                                latencyRoot,
                                partition.getString("dnn"),
                                partition.get("gpu"),
                                partition.getInt("mps") + 1));

        var mappingDnn = mappingRoot.toPath().resolve(partition.getString("dnn"));
        @Cleanup var transSizeFileGlob = Files.newDirectoryStream(
                mappingDnn, "contraction_trans-size_ilpprepart10-*.csv");
        var transSizeFile = transSizeFileGlob.iterator().next().toFile();

        IlpPartitionedModel m =
                new IlpPartitionedModel(
                        latencyFile,
                        transSizeFile,
                        partition.getJSONArray("layers").getInt(0),
                        partition.getJSONArray("layers").getInt(1));

        // Verify that model and plan runtime matches
        int latInferModel = m.getTotalLatency(partition.getInt("bs"));
        int latInferPlan = partition.getInt("lat_infer");
        int epsilon = 50; // microseconds
        assert Math.abs(latInferModel - latInferPlan) < epsilon : String.format(
                "latency mismatch: model %d, plan %d",
                latInferModel,
                latInferPlan);
        return m;
    }

    float getXput(int bs) {
        DoubleArrayList xputArr = new DoubleArrayList();
        for (int i = 0; i < models.size(); i++) {
            int lat = models.get(i).getTotalLatency(bs);
            float xput = bs * (float) 1e6 * workerss.get(i).size() / lat;
            xputArr.add(xput);
        }
        return (float) xputArr.doubleStream().min().getAsDouble();
    }
}
