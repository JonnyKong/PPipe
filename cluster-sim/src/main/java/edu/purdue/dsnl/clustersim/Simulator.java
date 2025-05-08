package edu.purdue.dsnl.clustersim;

import lombok.Getter;

import java.util.TreeSet;

public class Simulator {
    private static final Simulator SIMULATOR = new Simulator();
    private final TreeSet<Event> events = new TreeSet<>();

    @Getter
    private int time;

    public boolean containsEvent(Event e) {
        return events.contains(e);
    }

    public void addEvent(Event e) {
        events.add(e);
    }

    public void removeEvent(Event e) {
        events.remove(e);
    }

    public void doAllEvents() {
        while (true) {
            var e = events.pollFirst();
            if (e == null) {
                break;
            }
            time = e.time;
            e.execute();
        }
    }

    public static Simulator getInstance() {
        return SIMULATOR;
    }
}
