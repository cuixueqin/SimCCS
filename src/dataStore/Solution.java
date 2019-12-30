package dataStore;

import java.util.HashMap;
import java.util.HashSet;

/**
 *
 * @author yaw
 */
public class Solution {

    // Opened sources.
    private HashMap<Source, Double> sourceCaptureAmounts;   // MTCO2/yr
    private HashMap<Source, Double> sourceCosts;

    // Opened sinks.
    private HashMap<Sink, Double> sinkStorageAmounts;
    private HashMap<Sink, Integer> sinkNumWells;
    private HashMap<Sink, Double> sinkCosts;

    // Opened edges.
    private HashMap<Edge, Double> edgeTransportAmounts;
    private HashMap<Edge, Double> edgeCosts;
    private HashMap<Edge, Integer> edgeTrends;

    // Other.
    private double captureAmountPerYear;
    private int projectLength;
    private double crf;

    public Solution() {
        sourceCaptureAmounts = new HashMap<>();
        sourceCosts = new HashMap<>();
        sinkStorageAmounts = new HashMap<>();
        sinkCosts = new HashMap<>();
        edgeTransportAmounts = new HashMap<>();
        edgeCosts = new HashMap<>();
        sinkNumWells = new HashMap<>();
        edgeTrends = new HashMap<>();
    }

    public void addSourceCaptureAmount(Source src, double captureAmount) {
        if (!sourceCaptureAmounts.containsKey(src)) {
            sourceCaptureAmounts.put(src, 0.0);
        }
        sourceCaptureAmounts.put(src, sourceCaptureAmounts.get(src) + captureAmount);
        captureAmountPerYear += captureAmount;
    }

    public void addSourceCostComponent(Source src, double cost) {
        if (!sourceCosts.containsKey(src)) {
            sourceCosts.put(src, 0.0);
        }
        sourceCosts.put(src, sourceCosts.get(src) + cost);
    }

    public void addSinkStorageAmount(Sink snk, double captureAmount) {
        if (!sinkStorageAmounts.containsKey(snk)) {
            sinkStorageAmounts.put(snk, 0.0);
        }
        sinkStorageAmounts.put(snk, sinkStorageAmounts.get(snk) + captureAmount);
    }

    public void addSinkNumWells(Sink snk, int numWells) {
        if (!sinkNumWells.containsKey(snk)) {
            sinkNumWells.put(snk, 0);
        }
        sinkNumWells.put(snk, sinkNumWells.get(snk) + numWells);
    }

    public void addSinkCostComponent(Sink snk, double cost) {
        if (!sinkCosts.containsKey(snk)) {
            sinkCosts.put(snk, 0.0);
        }
        sinkCosts.put(snk, sinkCosts.get(snk) + cost);
    }

    public void addEdgeTransportAmount(Edge edg, double captureAmount) {
        if (!edgeTransportAmounts.containsKey(edg)) {
            edgeTransportAmounts.put(edg, 0.0);
        }
        edgeTransportAmounts.put(edg, edgeTransportAmounts.get(edg) + captureAmount);
    }

    public void setEdgeTrend(Edge edg, int trend) {
        edgeTrends.put(edg, trend);
    }

    public void addEdgeCostComponent(Edge edg, double cost) {
        if (!edgeCosts.containsKey(edg)) {
            edgeCosts.put(edg, 0.0);
        }
        edgeCosts.put(edg, edgeCosts.get(edg) + cost);
    }

    public void setSourceCaptureAmounts(HashMap<Source, Double> sourceCaptureAmounts) {
        this.sourceCaptureAmounts = sourceCaptureAmounts;
    }

    public void setSourceCosts(HashMap<Source, Double> sourceCosts) {
        this.sourceCosts = sourceCosts;
    }

    public void setSinkStorageAmounts(HashMap<Sink, Double> sinkStorageAmounts) {
        this.sinkStorageAmounts = sinkStorageAmounts;
    }

    public void setSinkCosts(HashMap<Sink, Double> sinkCosts) {
        this.sinkCosts = sinkCosts;
    }

    public void setEdgeTransportAmounts(HashMap<Edge, Double> edgeTransportAmounts) {
        this.edgeTransportAmounts = edgeTransportAmounts;
    }

    public void setEdgeCosts(HashMap<Edge, Double> edgeCosts) {
        this.edgeCosts = edgeCosts;
    }

    public void setSolutionCosts(DataStorer data) {
        for (Source src : sourceCaptureAmounts.keySet()) {
            double cost = src.getOpeningCost(crf) + src.getCaptureCost() * sourceCaptureAmounts.get(src);
            sourceCosts.put(src, cost);
        }

        for (Sink snk : sinkStorageAmounts.keySet()) {
            double cost = snk.getOpeningCost(crf);
            if (sinkNumWells.get(snk) != null) {
                cost += snk.getWellOpeningCost(crf) * sinkNumWells.get(snk);
            }
            cost += snk.getInjectionCost() * sinkStorageAmounts.get(snk);
            sinkCosts.put(snk, cost);
        }

        for (Edge edg : edgeTransportAmounts.keySet()) {
            LinearComponent[] linearComponents = data.getLinearComponents();
            HashMap<Edge, Double> edgeConstructionCosts = data.getGraphEdgeConstructionCosts();
            HashMap<Edge, Double> edgeRightOfWayCosts = data.getGraphEdgeRightOfWayCosts();

            double transportAmount = edgeTransportAmounts.get(edg);
            LinearComponent comp = null;
            boolean found = false;
            for (LinearComponent c : linearComponents) {
                if (!found && transportAmount <= c.getMaxCapacity()) {
                    found = true;
                    comp = c;
                }
            }

            double fixed = (comp.getConIntercept() * edgeConstructionCosts.get(edg) + comp.getRowIntercept() * edgeRightOfWayCosts.get(edg)) * crf;
            double variable = (comp.getConSlope() * edgeConstructionCosts.get(edg) + comp.getRowSlope() * edgeRightOfWayCosts.get(edg)) * crf / 1.0;
            double cost = fixed + variable * edgeTransportAmounts.get(edg);
            edgeCosts.put(edg, cost);
            
            if (edg.v1 == 16446255 && edg.v2 == 17112348) {
                System.out.println(comp.getMaxCapacity() + ", " + cost + ", " + edgeTransportAmounts.get(edg));
            }
        }
    }

    //public void setTargetCaptureAmountPerYear(double targetCaptureAmount) {
    //    this.captureAmountPerYear = targetCaptureAmount;
    //}
    public void setProjectLength(int projectLength) {
        this.projectLength = projectLength;
    }

    public void setCRF(double crf) {
        this.crf = crf;
    }

    public HashSet<Source> getOpenedSources() {
        return new HashSet<>(sourceCaptureAmounts.keySet());
    }

    public HashSet<Sink> getOpenedSinks() {
        return new HashSet<>(sinkStorageAmounts.keySet());
    }

    public HashSet<Edge> getOpenedEdges() {
        return new HashSet<>(edgeTransportAmounts.keySet());
    }

    public HashMap<Source, Double> getSourceCaptureAmounts() {
        return sourceCaptureAmounts;
    }

    public HashMap<Source, Double> getSourceCosts() {
        return sourceCosts;
    }

    public HashMap<Sink, Double> getSinkStorageAmounts() {
        return sinkStorageAmounts;
    }

    public HashMap<Sink, Double> getSinkCosts() {
        return sinkCosts;
    }

    public HashMap<Edge, Double> getEdgeTransportAmounts() {
        return edgeTransportAmounts;
    }

    public HashMap<Edge, Double> getEdgeCosts() {
        return edgeCosts;
    }

    public int getNumOpenedSources() {
        return sourceCaptureAmounts.keySet().size();
    }

    public int getNumOpenedSinks() {
        return sinkStorageAmounts.keySet().size();
    }

    public double getCaptureAmount() {
        double amountCaptured = 0;
        for (Source src : sourceCaptureAmounts.keySet()) {
            amountCaptured += sourceCaptureAmounts.get(src);
        }
        return amountCaptured * projectLength;
    }

    public double getAnnualCaptureAmount() {
        return captureAmountPerYear;
    }

    public int getNumEdgesOpened() {
        return edgeTransportAmounts.keySet().size();
    }

    public int getProjectLength() {
        return projectLength;
    }

    public double getCRF() {
        return crf;
    }

    public double getTotalAnnualCaptureCost() {
        double cost = 0;
        for (Source src : sourceCosts.keySet()) {
            cost += sourceCosts.get(src);
        }
        return cost;
    }

    public double getUnitCaptureCost() {
        if (captureAmountPerYear == 0) {
            return 0;
        }
        return getTotalAnnualCaptureCost() / captureAmountPerYear;
    }

    public double getTotalAnnualStorageCost() {
        double cost = 0;
        for (Sink snk : sinkCosts.keySet()) {
            cost += sinkCosts.get(snk);
        }
        return cost;
    }

    public double getUnitStorageCost() {
        if (captureAmountPerYear == 0) {
            return 0;
        }
        return getTotalAnnualStorageCost() / captureAmountPerYear;
    }

    public double getTotalAnnualTransportCost() {
        double cost = 0;
        for (Edge edg : edgeCosts.keySet()) {
            cost += edgeCosts.get(edg);
        }
        return cost;
    }

    public double getUnitTransportCost() {
        if (captureAmountPerYear == 0) {
            return 0;
        }
        return getTotalAnnualTransportCost() / captureAmountPerYear;
    }

    public double getTotalCost() {
        return getTotalAnnualCaptureCost() + getTotalAnnualStorageCost() + getTotalAnnualTransportCost();
    }

    public double getUnitTotalCost() {
        return getUnitCaptureCost() + getUnitStorageCost() + getUnitTransportCost();
    }

    public double getPercentCaptured(Source source) {
        return sourceCaptureAmounts.get(source) / source.getProductionRate();
    }

    public double getPercentStored(Sink sink) {
        return (sinkStorageAmounts.get(sink) * projectLength) / sink.getCapacity();
    }
}
