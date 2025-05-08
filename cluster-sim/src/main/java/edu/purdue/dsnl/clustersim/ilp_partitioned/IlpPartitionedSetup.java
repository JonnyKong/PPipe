package edu.purdue.dsnl.clustersim.ilp_partitioned;

import edu.purdue.dsnl.clustersim.*;

import org.json.JSONArray;

import picocli.CommandLine;

import java.io.File;
import java.nio.file.Files;
import java.util.concurrent.Callable;

@CommandLine.Command(name = "ilp_partitioned")
public class IlpPartitionedSetup extends Main.ParentOptions implements Callable<Integer> {

    @CommandLine.Option(names = { "--logdir" })
    public File logDir;

    @CommandLine.Option(names = {
            "--latency-root" }, defaultValue = "/export2/kong102/clusterserving_results/layer-timing")
    public File latencyRoot;

    @CommandLine.Option(names = { "--mapping-root" }, defaultValue = "../scripts/contraction_mappings")
    public File mappingRoot;

    @CommandLine.Option(names = { "--dnn-name" })
    public String dnnName;

    @CommandLine.Option(names = { "--json-plan-path" }, required = true)
    public File json_plan_path;

    @CommandLine.Option(names = "--dnn-id")
    public int dnnId = 0;

    @CommandLine.Option(names = "--lb")
    LbChoice lbChoice = LbChoice.RESERVATION;

    public enum LbChoice {
        SLA_AWARE,
        RESERVATION,
    }

    @Override
    public Integer call() throws Exception {
        String content = new String(Files.readAllBytes(json_plan_path.toPath()));
        JSONArray plan = new JSONArray(content);

        var clusterCfg = plan.getJSONObject(dnnId);
        Cluster c = new Cluster(clusterCfg, latencyRoot, mappingRoot, lbChoice);

        RequestGenerator gen;
        if (tracePath.isPresent()) {
            gen = new TraceRequestGenerator(c.lb, load, duration, (int) clusterCfg.getDouble("sla"),
                    tracePath.get());
        } else {
            gen = new PoissonRequestGenerator(c.lb, load, duration, (int) clusterCfg.getDouble("sla"));
        }

        Simulator.getInstance().doAllEvents();
        if (logDir != null) {
            logDir.mkdirs();
            c.lb.writeLog(new File(logDir, "master.csv"));
            for (int i = 0; i < c.pipelines.size(); i++) {
                for (int j = 0; j < c.pipelines.get(i).lbs.size(); j++) {
                    c.pipelines
                            .get(i).lbs
                            .get(j)
                            .writeLog(new File(logDir, String.format("%d-%d.csv", i, j)));
                }
            }
        }
        return 0;
    }
}
