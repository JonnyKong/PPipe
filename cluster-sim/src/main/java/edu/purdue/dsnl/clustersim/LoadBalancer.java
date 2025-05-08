package edu.purdue.dsnl.clustersim;

import com.fasterxml.jackson.dataformat.csv.CsvMapper;

import lombok.Cleanup;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

public abstract class LoadBalancer extends Event implements RequestsAcceptable {

    private static int idCounter = 0;

    public int id = idCounter++;

    protected final List<Request> queue = new ArrayList<>();

    protected final List<Request> log = new ArrayList<>();

    @Override
    public void acceptRequests(List<Request> requests) {
        Simulator.getInstance().removeEvent(this);
        this.time = Simulator.getInstance().getTime();
        for (Request r : requests) {
            r.timeArriveLb = time;
            r.queueLen = queue.size();
            queue.add(r);
        }
        queue.sort(Comparator.comparingInt(r -> r.deadline));
        execute();
    }

    public void writeLog(File logFile) throws IOException {
        var mapper = new CsvMapper();
        var schema = mapper.schemaFor(Request.class).withHeader();
        @Cleanup var writer = mapper.writer(schema).writeValues(logFile);
        writer.writeAll(log);
    }
}
