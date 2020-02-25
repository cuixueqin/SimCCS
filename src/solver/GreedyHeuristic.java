 /*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package solver;

import dataStore.DataStorer;
import dataStore.Edge;
import dataStore.HeuristicEdge;
import dataStore.Sink;
import dataStore.Source;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.PriorityQueue;
import java.util.Set;
import static utilities.Utilities.convertIntegerArray;


/**
 *
 * @author yaw
 */
public class GreedyHeuristic {

    private DataStorer data;

    private Source[] sources;
    private Sink[] sinks;

    // Graph
    private int[] graphVertices;
    private HeuristicEdge[][] adjacencyMatrix;
    private double[][] adjacencyCosts;
    private HashMap<Integer, Integer> cellNumToVertexNum;
    private HashMap<Integer, HashSet<Integer>> neighbors;
    
    private HashSet<Source> solutionSources;
    private HashSet<Sink> solutionSinks;
    

    public GreedyHeuristic(DataStorer data) {
        
        solutionSources = new HashSet<Source>();
        solutionSinks = new HashSet<Sink>();
        
        this.data = data;

        sources = data.getSources();
        sinks = data.getSinks();
        graphVertices = data.getGraphVertices();

        cellNumToVertexNum = new HashMap<>();
        neighbors = new HashMap<>();
    }
    
    
    // Every iteration, the algorithm will choose numPairs pairs with the lowest cost to add to the network.
    // Should be between 1 and the number of pairs (#srcs*#snks).
    public void solve(int numPairs, int modelVersion) {

        //Set number of pairs to srcs.length*snks.length if it is larger.
        numPairs = Math.min(numPairs, sources.length * sinks.length);

        // Initialize sources and sinks
        for (Source src : sources) {
            src.setRemainingCapacity(src.getProductionRate());
        }
        for (Sink snk : sinks) {
            snk.setRemainingCapacity(snk.getCapacity() / data.getProjectLength());
        }

        // Make directed edge graph
        Set<Edge> originalEdges = data.getGraphEdgeCosts().keySet();
        adjacencyMatrix = new HeuristicEdge[graphVertices.length][graphVertices.length];
        adjacencyCosts = new double[graphVertices.length][graphVertices.length];

        for (int u = 0; u < graphVertices.length; u++) {
            cellNumToVertexNum.put(graphVertices[u], u);
            for (int v = 0; v < graphVertices.length; v++) {
                adjacencyCosts[u][v] = Double.MAX_VALUE;
                if (originalEdges.contains(new Edge(graphVertices[u], graphVertices[v]))) {
                    if (neighbors.get(u) == null) {
                        neighbors.put(u, new HashSet<>());
                    }
                    neighbors.get(u).add(v);
                    adjacencyMatrix[u][v] = new HeuristicEdge(graphVertices[u], graphVertices[v], data);
                    adjacencyMatrix[u][v].currentHostingAmount = 0;
                    adjacencyMatrix[u][v].currentSize = 0;

                    adjacencyMatrix[v][u] = new HeuristicEdge(graphVertices[v], graphVertices[u], data);
                    adjacencyMatrix[v][u].currentHostingAmount = 0;
                    adjacencyMatrix[v][u].currentSize = 0;
                }
            }
        }
        

        if (modelVersion == 1) {
            capacityModel(numPairs);
        } else if (modelVersion == 2) {
            priceModel(numPairs);
        }
    }

    public void capacityModel(int numPairs) {
        long startTime = System.nanoTime();
        double amountCaptured = 0;  // Amount of CO2 currently captured/injected by algorithm
        
        //Total amount of CO2 storage possible from sources and sinks.
        double tolerance = 0.00000001; //Will capture amountPossible-tolerance. 
        double srcPossible=0;
        for (int srcNum = 0; srcNum < sources.length; srcNum++) {
        	srcPossible += sources[srcNum].getRemainingCapacity();
        }
        
        double snkPossible=0;
        for (int snkNum = 0; snkNum < sinks.length; snkNum++) {
            snkPossible += sinks[snkNum].getRemainingCapacity();
        }
        
        double amountPossible = Math.min(srcPossible,snkPossible)-tolerance;
        
        while (amountCaptured < data.getTargetCaptureAmount() && amountCaptured < amountPossible) {
        	
            // Make cost array
            Pair[][] pairCosts = makePairwiseCostArray(Math.min(data.getTargetCaptureAmount() - amountCaptured,
            													amountPossible - amountCaptured));

            // TODO: Look at making this more efficient.Probably return pairCosts initially.
            ArrayList<Pair> pairCostsList = new ArrayList<Pair>();
            for (int srcNum = 0; srcNum < sources.length; srcNum++) {
                for (int snkNum = 0; snkNum < sinks.length; snkNum++) {
                    pairCostsList.add(pairCosts[srcNum][snkNum]);
                }
            }

            pairCostsList.sort(new PairComparator());

            Pair cheapest[] = new Pair[numPairs];
            cheapest = pairCostsList.subList(0, numPairs).toArray(cheapest);
            
            double transferAmount = 0;
            for (int i = 0; i < cheapest.length; i++) {
                
                
                transferAmount = Math.min(Math.min(Math.min(
                		cheapest[i].src.getRemainingCapacity(), 
                		cheapest[i].snk.getRemainingCapacity()),
                		data.getTargetCaptureAmount() - amountCaptured),
                		amountPossible - amountCaptured);
                
                amountCaptured += transferAmount;
                
                
                schedulePair(cheapest[i].src, cheapest[i].snk, cheapest[i].path, transferAmount);

                if (amountCaptured >= data.getTargetCaptureAmount() || amountCaptured >= amountPossible) {
                    break;
                }
            }
            System.out.println("Captured " + amountCaptured + " of " + data.getTargetCaptureAmount());
        }
        System.out.println("Execution Time: " + (System.nanoTime() - startTime) / 1000000000 + " seconds");
    }

    public void priceModel(int numPairs) {
        long startTime = System.nanoTime();

        boolean negativePair = true;

        while (negativePair) {
            // Make cost array
            Pair[][] pairCosts = makePairwiseCostArray(Double.MAX_VALUE);

            // TODO: Look at making this more efficient. Probably return pairCosts initially.
            ArrayList<Pair> pairCostsList = new ArrayList<Pair>();
            for (int srcNum = 0; srcNum < sources.length; srcNum++) {
                for (int snkNum = 0; snkNum < sinks.length; snkNum++) {
                    pairCostsList.add(pairCosts[srcNum][snkNum]);

                }
            }

            pairCostsList.sort(new PairComparator());

            Pair cheapest[] = new Pair[numPairs];
            cheapest = pairCostsList.subList(0, numPairs).toArray(cheapest);
            
            if (cheapest[0].cost >= 0) {
                negativePair = false;
            }
            

            double transferAmount = 0;
            for (int i = 0; i < cheapest.length; i++) {
                
                
                if (cheapest[i].cost < 0) {
                    transferAmount = Math.min(cheapest[i].src.getRemainingCapacity(), cheapest[i].snk.getRemainingCapacity());
                    schedulePair(cheapest[i].src, cheapest[i].snk, cheapest[i].path, transferAmount);
                }
            }
        }
        System.out.println("Execution Time: " + (System.nanoTime() - startTime) / 1000000000 + " seconds");
    }

    public void schedulePair(Source src, Sink snk, HashSet<HeuristicEdge> path, double transferAmount) {

        src.setRemainingCapacity(src.getRemainingCapacity() - transferAmount);
        snk.setRemainingCapacity(snk.getRemainingCapacity() - transferAmount);
        
        double totalTransferAmount = snk.getCapacity() / data.getProjectLength() - snk.getRemainingCapacity();
        snk.setNumWells(getNewNumWells(snk, totalTransferAmount));

        for (HeuristicEdge frontEdge : path) {
            HeuristicEdge backEdge = adjacencyMatrix[cellNumToVertexNum.get(frontEdge.v2)][cellNumToVertexNum.get(frontEdge.v1)];

            // If edge in opposite direction was hosting flow
            if (backEdge.currentHostingAmount > 0) {
                // If the back edge is still needed
                if (transferAmount < backEdge.currentHostingAmount) {
                    // Calculate the new pipeline size
                    int newSize = getNewPipelineSize(backEdge, backEdge.currentHostingAmount - transferAmount);

                    // Update pipeline size
                    backEdge.currentSize = newSize;

                    // Update hosting amount
                    backEdge.currentHostingAmount -= transferAmount;
                } else if (transferAmount > backEdge.currentHostingAmount) {    //If front edge is now needed
                    int newSize = getNewPipelineSize(frontEdge, transferAmount - backEdge.currentHostingAmount);
                    frontEdge.currentSize = newSize;
                    frontEdge.currentHostingAmount = transferAmount - backEdge.currentHostingAmount;
                    
                    backEdge.currentSize = 0;
                    backEdge.currentHostingAmount = 0;
                } else {
                    backEdge.currentSize = 0;
                    backEdge.currentHostingAmount = 0;
                }
            } else {
                int newSize = getNewPipelineSize(frontEdge, transferAmount + frontEdge.currentHostingAmount);
                frontEdge.currentSize = newSize;
                frontEdge.currentHostingAmount += transferAmount;
            }
        }
    }

    public Pair[][] makePairwiseCostArray(double remainingCaptureAmount) {
        Pair[][] pairCosts = new Pair[sources.length][sinks.length];
        for (int srcNum = 0; srcNum < sources.length; srcNum++) {
            for (int snkNum = 0; snkNum < sinks.length; snkNum++) {
                Source src = sources[srcNum];
                Sink snk = sinks[snkNum];

                double transferAmount = Math.min(Math.min(src.getRemainingCapacity(), snk.getRemainingCapacity()), remainingCaptureAmount);
                double cost = Double.MAX_VALUE;
                HashSet<HeuristicEdge> path = null;

                if (transferAmount > 0) {
                    cost = 0;
                    // Incurr opening cost if source not yet used
                    if (src.getRemainingCapacity() == src.getProductionRate()) {
                        cost += src.getOpeningCost(data.getCrf());
                    }
                    cost += transferAmount * src.getCaptureCost();

                    // Incurr opening cost if sink not yet used
                    if (snk.getRemainingCapacity() == snk.getCapacity() / data.getProjectLength()) {
                        cost += snk.getOpeningCost(data.getCrf());
                    }
                    // Determine cost of additional wells needed
                    int numNewWells = getNewNumWells(snk, transferAmount) - snk.getNumWells();
                    cost += snk.getWellOpeningCost(data.getCrf()) * numNewWells;
                    cost += transferAmount * snk.getInjectionCost();

                    // Assign costs to graph
                    //setGraphCosts(src, snk, transferAmount);

                    // Find shortest path between src and snk
                    Object[] data = dijkstra(src, snk,transferAmount);
                    path = (HashSet<HeuristicEdge>) data[0];
                    double pathCost = (double) data[1];

                    cost += pathCost;

                    // Cost per ton of CO2
                    cost /= transferAmount;
                }

                pairCosts[srcNum][snkNum] = new Pair(src, snk, path, cost);
            }
        }
        return pairCosts;
    }

    public int getNewPipelineSize(HeuristicEdge edge, double volume) {
        double[] capacities = edge.capacities;
        int size = 0;
        while (volume > capacities[size]) {
            size++;
        }
        return size;
    }

    public int getNewNumWells(Sink snk, double volume) {
        return (int) Math.ceil(volume / snk.getWellCapacity());
    }

    // Dijkstra to run on graph edges
    public Object[] dijkstra(Source src, Sink snk,double transferAmount) {
        
        
        int srcVertexNum = cellNumToVertexNum.get(src.getCellNum());
        int snkVertexNum = cellNumToVertexNum.get(snk.getCellNum());

        int numNodes = graphVertices.length;
        PriorityQueue<GreedyHeuristic.Data> pQueue = new PriorityQueue<>(numNodes);
        double[] costs = new double[numNodes];
        int[] previous = new int[numNodes];
        GreedyHeuristic.Data[] map = new GreedyHeuristic.Data[numNodes];

        for (int vertex = 0; vertex < numNodes; vertex++) {
            costs[vertex] = Double.MAX_VALUE;
            previous[vertex] = -1;
            map[vertex] = new GreedyHeuristic.Data(vertex, costs[vertex]);
        }

        costs[srcVertexNum] = 0;
        map[srcVertexNum].distance = 0;
        pQueue.add(map[srcVertexNum]);

        while (!pQueue.isEmpty()) {
            GreedyHeuristic.Data u = pQueue.poll();
            for (int neighbor : neighbors.get(u.vertexNum)) {
                if (adjacencyMatrix[u.vertexNum][neighbor] != null) {
                    
                    
                    HeuristicEdge frontEdge = adjacencyMatrix[u.vertexNum][neighbor];
                    HeuristicEdge backEdge = adjacencyMatrix[neighbor][u.vertexNum];
                    double edgeCost = 0;

                    if (frontEdge != null) {
                        // If edge in opposite direction is hosting flow
                        if (backEdge.currentHostingAmount > 0) {
                            // Remove back edge (because it will need to change)
                            edgeCost -= backEdge.buildCost[backEdge.currentSize];
                            edgeCost -= backEdge.currentHostingAmount * backEdge.transportCost[backEdge.currentSize];

                            // If the back edge is still needed
                            if (transferAmount < backEdge.currentHostingAmount) {
                                // Calculate the new pipeline size
                                int newSize = getNewPipelineSize(backEdge, backEdge.currentHostingAmount - transferAmount);

                                // Factor in build costs
                                edgeCost += backEdge.buildCost[newSize];

                                // Factor in utilization costs
                                edgeCost += backEdge.transportCost[newSize] * (backEdge.currentHostingAmount - transferAmount);
                            } else if (transferAmount > backEdge.currentHostingAmount) {    //If front edge is now needed
                                int newSize = getNewPipelineSize(frontEdge, transferAmount - backEdge.currentHostingAmount);
                                edgeCost += frontEdge.buildCost[newSize];
                                edgeCost += frontEdge.transportCost[newSize] * (transferAmount - backEdge.currentHostingAmount);
                            }
                        } else {
                            int newSize = getNewPipelineSize(frontEdge, transferAmount + frontEdge.currentHostingAmount);
                            edgeCost += frontEdge.buildCost[newSize] - frontEdge.buildCost[frontEdge.currentSize];
                            edgeCost += frontEdge.transportCost[newSize] * (transferAmount + frontEdge.currentHostingAmount) - frontEdge.transportCost[frontEdge.currentSize] * (frontEdge.currentHostingAmount);
                        }
                    }
                    
                    double altDistance = costs[u.vertexNum] + Math.max(edgeCost, 0);
                    if (altDistance < costs[neighbor]) {
                        costs[neighbor] = altDistance;
                        previous[neighbor] = u.vertexNum;

                        map[neighbor].distance = altDistance;
                        pQueue.add(map[neighbor]);
                    }
                }
            }
        }

        HashSet<HeuristicEdge> path = new HashSet<>();
        int node = snkVertexNum;
        while (node != srcVertexNum) {
            int previousNode = previous[node];
            path.add(adjacencyMatrix[previousNode][node]);
            node = previousNode;
        }

        return new Object[]{path, costs[snkVertexNum]};
    }

    public Source[] getSources() {
        return sources;
    }

    public Sink[] getSinks() {
        return sinks;
    }

    public HashMap<Integer, HashSet<Integer>> getNeighbors() {
        return(neighbors);
    }
    
    public int[] getGraphVertices() {
        return graphVertices;
    }

    public HeuristicEdge[][] getAdjacencyMatrix() {
        return adjacencyMatrix;
    }

    public HashMap<Integer, Integer> getCellVertexMap() {
        return cellNumToVertexNum;
    }
    
    
    private class Data implements Comparable<Data> {

        public int vertexNum;
        public double distance;

        public Data(int cellNum, double distance) {
            this.vertexNum = cellNum;
            this.distance = distance;
        }

        @Override
        public int compareTo(Data other) {
            return Double.valueOf(distance).compareTo(other.distance);
        }

        @Override
        public int hashCode() {
            return vertexNum;
        }

        public boolean equals(Data other) {
            return distance == other.distance;
        }
    }

    private class PairComparator implements Comparator<Pair> {

        @Override
        public int compare(Pair arg0, Pair arg1) {
            return Double.compare(arg0.cost, arg1.cost);
        }
    }

    private class Pair {

        public HashSet<HeuristicEdge> path;
        public double cost;
        public Source src;
        public Sink snk;

        public Pair(Source src, Sink snk, HashSet<HeuristicEdge> path, double cost) {
            this.src = src;
            this.snk = snk;
            this.path = path;
            this.cost = cost;
        }
    }
}
