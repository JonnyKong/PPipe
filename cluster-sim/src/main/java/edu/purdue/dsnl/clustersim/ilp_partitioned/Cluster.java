package edu.purdue.dsnl.clustersim.ilp_partitioned;

import edu.purdue.dsnl.clustersim.LoadBalancer;
import edu.purdue.dsnl.clustersim.Worker;
import edu.purdue.dsnl.clustersim.ilp_partitioned.IlpPartitionedSetup.LbChoice;
import org.json.JSONObject;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

class Cluster {
    public List<Pipeline> pipelines;

    public LoadBalancer lb;

    public Cluster(JSONObject cfg, File latencyRoot, File mappingRoot, LbChoice lbChoice) {
        int bandwidth = cfg.getJSONObject("config").getInt("bw_gbps");
        pipelines = new ArrayList<Pipeline>();
        for (int i = 0; i < cfg.getJSONArray("pipelines").length(); i++) {
            JSONObject pipeline_cfg = cfg.getJSONArray("pipelines").getJSONObject(i);
            pipelines.add(new Pipeline(pipeline_cfg, latencyRoot, mappingRoot, bandwidth, lbChoice));
        }

        int sla = cfg.getInt("sla");

        if (lbChoice == LbChoice.SLA_AWARE) {
            lb = new SlaAwareLoadBalancer(getFirstPartWorkers());
        } else if (lbChoice == LbChoice.RESERVATION) {
            lb = new ReservationSlaLoadBalancer(pipelines, sla);
        }
    }

    /** Return the list of workers that are in the first part of each pipeline. */
    public List<Worker> getFirstPartWorkers() {
        ArrayList<Worker> ret = new ArrayList<>();
        for (Pipeline p : pipelines) {
            ret.addAll(p.workerss.get(0));
        }
        return ret;
    }
}
