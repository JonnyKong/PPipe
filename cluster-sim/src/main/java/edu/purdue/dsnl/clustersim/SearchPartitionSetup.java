/**
 * Given the SLO, the number of GPUs, and the runtimes for each DNN partition,
 * use heurstics to search for how many GPUs to assign to each partition, and
 * the SLA for each.
 *
 * Enumerate all possible assignments of GPUs to DNN parts. For each assignment,
 * adjust the SLO of each part to make the SLO equal across all parts.
 */
package edu.purdue.dsnl.clustersim;

import edu.purdue.dsnl.clustersim.manual_partitioned.ManualPartitionedModel;
import picocli.CommandLine;

import java.io.File;
import java.util.*;
import java.util.concurrent.Callable;
import java.util.stream.IntStream;

@CommandLine.Command(name = "search_partition")
public class SearchPartitionSetup implements Callable<Integer> {
    @CommandLine.Option(names={"-w", "--worker"})
    int worker;

    @CommandLine.Option(names={"-s", "--slo"})
    int slo;

    @CommandLine.Parameters(paramLabel="latfiles")
    File[] latFiles;

    int[] mpsMultipliers = new int[]{1, 1};

    List<ManualPartitionedModel> models;

    @Override
    public Integer call() throws Exception {
        assert latFiles.length == mpsMultipliers.length;

        models = ManualPartitionedModel.getPartitionedModels(latFiles);

        List<List<Integer>> workerParts = getWorkerParts();
        // List<List<Integer>> workerParts = getWorkerPartsV2();

        for (List<Integer> workerPart : workerParts) {
            // System.out.println();
            // System.out.println("workerPartition: " + workerPart);
            List<Integer> sloPart = adjustSlo(workerPart);
            double tput = IntStream.range(0, models.size())
                    .mapToDouble(i -> computeTput(models.get(i), sloPart.get(i), workerPart.get(i),
                                mpsMultipliers[i]))
                    .min().getAsDouble();
            System.out.format("\"%s\",\"%s\",%f\n",
                              workerPart.toString(), sloPart.toString(), tput);
        }

        return 0;
    }

    public List<List<Integer>> getWorkerParts() {
        List<List<Integer>> ret = new ArrayList<>();
        getWorkerPartsHelper(new LinkedList<Integer>(), worker, ret);
        return ret;
    }

    public List<List<Integer>> getWorkerPartsV2() {
        List<List<Integer>> ret = new ArrayList<>();
        for (int i = 1; i <= 15; i++) {
            for (int j = 1; j <= 32; j++) {
                ret.add(List.of(i, j, 16 - i));
            }
        }
        return ret;
    }

    public void getWorkerPartsHelper(List<Integer> workerParts, int workersRemaining,
                                          List<List<Integer>> ret) {
        if (workerParts.size() == models.size() - 1) {
            // Make last stage use up all remaining workers
            workerParts.add(workersRemaining);
            ret.add(new ArrayList<>(workerParts));
            workerParts.remove(workerParts.size() - 1);

        } else {
            // Assign minimum 1 worker to this part, and max servers such that it leaves at
            // least 1 worker for each remaining stage
            int remainingParts = models.size() - workerParts.size() - 1;

            for (int i = 1; i <= workersRemaining - remainingParts; i++) {
                workerParts.add(i);
                getWorkerPartsHelper(workerParts, workersRemaining - i, ret);
                workerParts.remove(workerParts.size() - 1);
            }
        }
    }

    private List<Integer> adjustSlo(List<Integer> workerPart) {
        int[] sloPart = new int[models.size()];
        int[] sloMin = new int[models.size()]; // SLO for BS=1

        // Start with equal slo to each partition
        for (int i = 0; i < models.size(); i++) {
            sloPart[i] = slo / models.size();
            sloMin[i] = models.get(i).getTotalLatency(1);
        }
        // System.out.println("sloMin: " + Arrays.toString(sloMin));

        int[] sloPartBest = null;
        double tputBest = 0.0;
        int adjustAmt = slo / models.size() / 2;
        int prevMinPart = -1;
        int prevMaxPart = -1;

        while (adjustAmt > 0) {
            double[] tputs = IntStream.range(0, models.size())
                    .mapToDouble(i -> computeTput(models.get(i), sloPart[i], workerPart.get(i),
                                mpsMultipliers[i]))
                    .toArray();
            double tput = Arrays.stream(tputs).min().getAsDouble();
            if (tput > tputBest) {
                tputBest = tput;
                sloPartBest = sloPart.clone();
            }
            // System.out.println("sloPart: " + Arrays.toString(sloPart));
            // System.out.println("tputs: " + Arrays.toString(tputs));

            // Find the part with min and max throughput. When searching for the max throughput part, exclude
            // the parts that are already assigned with min possible SLO
            int[] sortedIndices = IntStream.range(0, tputs.length)
                .boxed().sorted((i, j) -> (int) tputs[i] - (int) tputs[j])
                .mapToInt(ele -> ele).toArray();
            int minPart = sortedIndices[0];
            int maxPart = sortedIndices[sortedIndices.length - 1];
            for (int i = sortedIndices.length - 1; i >= 0; i--) {
                maxPart = sortedIndices[i];
                if (sloPart[maxPart] != sloMin[maxPart]) {
                    break;
                }
            }
            
            if (minPart == maxPart) {
                break;
            } else if (tputs[maxPart] - tputs[minPart] < tputs[maxPart] * 0.005) {
                break;
            }

            // Shrink adjustAmt on min or max partition change
            if (minPart != prevMinPart && maxPart != prevMaxPart) {
                adjustAmt *= 0.75;
            }
            prevMinPart = minPart;
            prevMaxPart = maxPart;

            // Adjust less than adjustAmt if bounded by minSlo
            if (sloPart[maxPart] - adjustAmt < sloMin[maxPart]) {
                sloPart[minPart] += (sloPart[maxPart] - sloMin[maxPart]);
                sloPart[maxPart] -= (sloPart[maxPart] - sloMin[maxPart]);
            } else {
                sloPart[minPart] += adjustAmt;
                sloPart[maxPart] -= adjustAmt;
            }
        }

        // System.out.println("best tput: " + tputBest);
        // System.out.println("best slo part: " + Arrays.toString(sloPartBest));
        return Arrays.stream(sloPartBest).boxed().toList();
    }

    private double computeTput(Model model, int slo, int numWorker, int mpsMultiplier) {
        int bs = model.maxBatchSize(slo);
        if (bs == 0) {
            return 0.0;
        } else {
            // latency is in us, mult 1e6 to convert tput to per second
            return (double) numWorker * bs * mpsMultiplier / model.getTotalLatency(bs) * 1e6;
        }
    }
}