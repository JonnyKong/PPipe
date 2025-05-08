package edu.purdue.dsnl.clustersim;

import edu.purdue.dsnl.clustersim.ilp_partitioned.IlpPartitionedSetup;
import edu.purdue.dsnl.clustersim.manual_partitioned.ManualPartitionedSetup;
import edu.purdue.dsnl.clustersim.s3dnn.S3dnnSetup;

import picocli.CommandLine;

import java.io.File;
import java.util.Optional;

@CommandLine.Command(subcommands = {
        SimpleSetup.class,
        S3dnnSetup.class,
        ManualPartitionedSetup.class,
        IlpPartitionedSetup.class,
        SearchPartitionSetup.class,
})
public class Main {
    public static class ParentOptions {
        @CommandLine.Option(names = { "-t", "--load" })
        public int load = 400;

        @CommandLine.Option(names = {
                "--trace-path" }, description = "Path to request arrival trace in MAF '21 format. If specified, requests"
                        + " will be issued by scaling up the trace `load` qps. If not"
                        + " specified, requests will be issued in Poisson arrival")
        public Optional<String> tracePath;

        @CommandLine.Option(names = { "-d", "--duration" })
        public int duration = 30000000;

        @CommandLine.Option(names = { "-o", "--slo" })
        public int slo = 60000;

        @CommandLine.Option(names = { "-l", "--log" })
        public File logFile;
    }

    public static void main(String[] args) {
        System.exit(new CommandLine(new Main()).execute(args));
    }
}