package edu.purdue.dsnl.clustersim;

import lombok.EqualsAndHashCode;

@EqualsAndHashCode
public abstract class Event implements Comparable<Event> {
    private static int uidGen = 0;

    private final int uid = uidGen++;

    protected int time;

    public abstract void execute();

    @Override
    public int compareTo(Event o) {
        if (time != o.time) {
            return time - o.time;
        }
        return uid - o.uid;
    }
}
