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
import dataStore.*;


/**
 *
 * @author yaw
 */
public class ImproveHeuristic {

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
    
    private double tolerance=Math.pow(10,-10);

    //TODO: Create validity function.
    //TODO: redo edge centraility stuff now that we have this bug fixed.
    public ImproveHeuristic(DataStorer data) {
        
        solutionSources = new HashSet<Source>();
        solutionSinks = new HashSet<Sink>();
        
        this.data = data;

        sources = data.getSources();
        sinks = data.getSinks();
        graphVertices = data.getGraphVertices();

        cellNumToVertexNum = new HashMap<>();
        neighbors = new HashMap<>();
    }
    
    public ImproveHeuristic(DataStorer data, Solution solution) {
        
        solutionSources = new HashSet<Source>();
        solutionSinks = new HashSet<Sink>();
        
        
        //1. Initialize from the data storer
        this.data = data;

        sources = data.getSources();
        sinks = data.getSinks();
        graphVertices = data.getGraphVertices();

        cellNumToVertexNum = new HashMap<>();
        neighbors = new HashMap<>();
        
        //2. Initialize solution.
        for(Source src:sources) {
            src.setRemainingCapacity(src.getProductionRate());
        }
        
        for(Sink snk: sinks) {
            snk.setRemainingCapacity(snk.getCapacity() / data.getProjectLength());
        }
        
        HashMap<Source,Double> sourceCaptureAmounts = solution.getSourceCaptureAmounts();
        for (Source src : solution.getOpenedSources()) {
            src.setRemainingCapacity(src.getProductionRate()-sourceCaptureAmounts.get(src));
            solutionSources.add(src);
        }
        
        HashMap<Sink,Double> sinkStorageAmounts = solution.getSinkStorageAmounts();
        for (Sink snk : solution.getOpenedSinks()) {
            snk.setRemainingCapacity(snk.getCapacity() / data.getProjectLength()-sinkStorageAmounts.get(snk));
            double totalTransferAmount = snk.getCapacity() / data.getProjectLength() - snk.getRemainingCapacity();
            snk.setNumWells(getNewNumWells(snk, totalTransferAmount));
            solutionSinks.add(snk);
        }

        // Make directed edge graph
        Set<Edge> originalEdges = data.getGraphEdgeCosts().keySet();
        adjacencyMatrix = new HeuristicEdge[graphVertices.length][graphVertices.length];
        adjacencyCosts = new double[graphVertices.length][graphVertices.length];
        
        HashMap<Edge,Double> edgeTransportAmounts= solution.getEdgeTransportAmounts();

        // First, set up initial edges.
         for (int u = 0; u < graphVertices.length; u++) {
            cellNumToVertexNum.put(graphVertices[u], u);
            for (int v = 0; v < graphVertices.length; v++) {
                adjacencyCosts[u][v] = Double.MAX_VALUE;
                Edge originalEdge = new Edge(graphVertices[u],graphVertices[v]);
                if (originalEdges.contains(originalEdge)) {
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
        
        // Now find edges and set their corresponding amount.
        for(Edge edge: edgeTransportAmounts.keySet()) {
            int u = cellNumToVertexNum.get(edge.v1);
            int v = cellNumToVertexNum.get(edge.v2);
            
            adjacencyMatrix[u][v].currentHostingAmount = edgeTransportAmounts.get(edge);
            adjacencyMatrix[u][v].currentSize = getNewPipelineSize(adjacencyMatrix[u][v], adjacencyMatrix[u][v].currentHostingAmount);
        }
        
        
    }

    public void improve(int iterations) {
        
        //Initialize things.
        //For iterations.
        
        //Choose Remove.
        //Remove.
        //Choose Add.
        //Add.
        
        for(int i =0; i<iterations;i++){
            
            //TODO: Think about smarter ways to remove and then add stuff in. Right now, the removal might end up always choosing longest paths.
            // Might need to divide the total removal by number of edges or something.
            double transferAmount = 0;
            // Remove solution.
            if(solutionSources.size()!=0 && solutionSinks.size()!=0) {
                Pair toRemove = chooseRemove();
                if(toRemove.path!=null) {
                    transferAmount = remove(toRemove);
                }
            }
            
            // Add solution.
            Pair toAdd = chooseAdd(transferAmount);
            
            System.out.println("Original transfer amount:"+transferAmount);
            System.out.println("Actual transfer amount:"+toAdd.amount);
            
            if(toAdd.path != null) {
                add(toAdd);
            }
            
            
        }
        
    }
    
    public Pair chooseAdd(double transferAmount) {
  
        ArrayList<Pair> pairCostsList = makePairwiseCostArray(transferAmount);

        pairCostsList.sort(new PairComparator());

        Pair cheapest;
        cheapest = pairCostsList.get(0); 


        System.out.println("Cheapest: Snk: "+cheapest.snk.getCellNum()+",Src: "+cheapest.src.getCellNum()+"PathCost: "+cheapest.cost);

        return cheapest;
    }
    
    public double add(Pair pair) {

        double transferAmount=pair.amount;
        
        Source src =  pair.src;
        Sink snk = pair.snk;
        
        
        src.setRemainingCapacity(src.getRemainingCapacity() - transferAmount);
        snk.setRemainingCapacity(snk.getRemainingCapacity() - transferAmount);
        
        
        solutionSources.add(src);
        solutionSinks.add(snk);

        double totalTransferAmount = snk.getCapacity() / data.getProjectLength() - snk.getRemainingCapacity();
        snk.setNumWells(getNewNumWells(snk, totalTransferAmount));

        for (HeuristicEdge frontEdge : pair.path) {
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
        
        
        return(transferAmount);
    }
    
    // Remove the src,snk, and pair from the solution set.
    public double remove(Pair pair) {

        double transferAmount=pair.amount;
        
        //Remove src amount.
        Source source =  pair.src;
        source.setRemainingCapacity(source.getRemainingCapacity()+transferAmount);
       
        
        //Only remove if we are set on capacity.
        if(source.getProductionRate()-source.getRemainingCapacity()<tolerance &&
           source.getProductionRate()-source.getRemainingCapacity()>-tolerance) {
           source.setRemainingCapacity(source.getProductionRate());
           solutionSources.remove(source);
        }
        System.out.println("Removed "+transferAmount+"from Source "+source.getLabel());
        
        //Remove snk amount.
        Sink sink = pair.snk;
        sink.setRemainingCapacity(sink.getRemainingCapacity()+transferAmount);
        
        //Only remove if we are set on capcity.
        if(sink.getCapacity()/data.getProjectLength() - sink.getRemainingCapacity()<tolerance &&
           sink.getCapacity()/data.getProjectLength() - sink.getRemainingCapacity() > -tolerance) {
           sink.setRemainingCapacity(sink.getCapacity()/data.getProjectLength());
           solutionSinks.remove(sink);
        }
        System.out.println("Removed "+transferAmount+"from Sink "+sink.getLabel());
       
        
        //Remove Edges.
        for(HeuristicEdge edge: pair.path) {
            
            // Update hosting amount
            edge.currentHostingAmount -= transferAmount;

            //Rounding error problem.
            if(edge.currentHostingAmount<tolerance &&
               edge.currentHostingAmount>-tolerance) {
                edge.currentHostingAmount=0;
            }

            int newSize = getNewPipelineSize(edge,edge.currentHostingAmount - transferAmount);
            
            // Update pipeline size
            edge.currentSize = newSize;

        }        
        
        return transferAmount;
    }
    
    public Pair chooseRemove() {
    
            //Initialize things.
            //For iterations.
            
            ArrayList<Pair> pairCostsList = makePairwiseCostRemovalArray();

            pairCostsList.sort(new PairComparator());

            Pair expensivest;
            expensivest = pairCostsList.get(pairCostsList.size()-1); 
            
//            for(Pair pair: pairCostsList) {
//                if(!Double.isInfinite(expensivest.cost)) {
//                    System.out.println(pair.cost);
//                }
//            }
            
            System.out.println("Expensive: Snk:"+expensivest.snk.getCellNum()+",Src:"+expensivest.src.getCellNum()+"PathCost:"+expensivest.cost);
            
            return expensivest;
    }
    
    //Here, we look at the cost of removing a flow.
    public ArrayList<Pair> makePairwiseCostRemovalArray() {
        
        ArrayList<Pair> pairCosts = new ArrayList<Pair>(solutionSources.size()+solutionSinks.size());
        
        for(Source src: solutionSources) {
            for(Sink snk: solutionSinks) {
               
                 
                //Modify this for perhaps better performance. Could reach optimal performance anywhere from 0 to max capacity.
                double snkUsed = snk.getCapacity()/ data.getProjectLength()-snk.getRemainingCapacity();
                double srcUsed = src.getProductionRate()-src.getRemainingCapacity();
                double transferAmount = Math.min(snkUsed,srcUsed);
                
                double cost = Double.MAX_VALUE;
                HashSet<HeuristicEdge> path = null;

                if (transferAmount > 0) {
                    cost = 0;
                    
                    // Incurr opening cost if removing this flow will deplete the entire source.
                    if (transferAmount==srcUsed) {
                        cost += src.getOpeningCost(data.getCrf());
                    }
                    cost += transferAmount * src.getCaptureCost();

                    // Incurr opening cost if removing this flow will deplete the entire sink.
                    if (transferAmount==snkUsed) {
                        cost += snk.getOpeningCost(data.getCrf());
                    }
                    
                    // Determine cost of additional wells needed
                    int numNewWells = getNewNumWells(snk, transferAmount) - snk.getNumWells();
                    cost += snk.getWellOpeningCost(data.getCrf()) * numNewWells;
                    
                    cost += transferAmount * snk.getInjectionCost();
                    
                    // Assign costs to graph
                    setRemovalGraphCosts(src, snk, transferAmount);

                    // Find shortest path between src and snk
                    Object[] data = removeDijkstra(src, snk,transferAmount);
                    path = (HashSet<HeuristicEdge>) data[0];
                    //double pathCost = (double) data[1];
                    
                    double pathCost=0;
                    if(path!=null) {
                       
                        for(HeuristicEdge edge: path) {

                            //Remove all costs.
                            pathCost += edge.buildCost[edge.currentSize];
                            pathCost += edge.currentHostingAmount * edge.transportCost[edge.currentSize];

                            int newSize = getNewPipelineSize(edge, edge.currentHostingAmount - transferAmount);

                            // Factor in build costs
                            pathCost -= edge.buildCost[newSize];

                            // Factor in utilization costs
                            pathCost -= edge.transportCost[newSize] * (edge.currentHostingAmount - transferAmount);

                        }
                    } else {
                        //If the path doesn't exist, then we want it's cost to be as low as possible so that it isn't chosen.
                        pathCost=Double.NEGATIVE_INFINITY;
                    }
                    

                    cost += pathCost;

                    // Cost per ton of CO2
                    cost /= transferAmount;
                }

                pairCosts.add(new Pair(src, snk, path, cost,transferAmount));
            }
        }
        return pairCosts;
    }

    public void schedulePair(Source src, Sink snk, HashSet<HeuristicEdge> path, double transferAmount) {

        src.setRemainingCapacity(src.getRemainingCapacity() - transferAmount);
        snk.setRemainingCapacity(snk.getRemainingCapacity() - transferAmount);
        
        
        solutionSources.add(src);
        solutionSinks.add(snk);

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

    public ArrayList<Pair> makePairwiseCostArray(double remainingCaptureAmount) {
        ArrayList<Pair> pairCosts = new ArrayList<Pair>(sources.length*sinks.length);
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
                    //TODO: Figure out what's going on here. Ask if commenting that out was what I needed to actually do. Assuming that we are looking at cost to remove those.
                    int numNewWells = getNewNumWells(snk, transferAmount); //- snk.getNumWells();
                    cost += snk.getWellOpeningCost(data.getCrf()) * numNewWells;
                    cost += transferAmount * snk.getInjectionCost();

                    // Assign costs to graph
                    setGraphCosts(src, snk, transferAmount);

                    // Find shortest path between src and snk
                    Object[] data = dijkstra(src, snk);
                    path = (HashSet<HeuristicEdge>) data[0];
                    double pathCost = (double) data[1];
                    
                    cost += pathCost;

                    // Cost per ton of CO2
                    cost /= transferAmount;
                }

                pairCosts.add(new Pair(src, snk, path, cost,transferAmount));
            }
        }
        return pairCosts;
    }
    
    // For a given src/snk pair, set the cost of the edgs to carry transferAmount of CO2
    public void setGraphCosts(Source src, Sink snk, double transferAmount) {
        for (int u = 0; u < graphVertices.length; u++) {
            for (int v = 0; v < graphVertices.length; v++) {
                HeuristicEdge frontEdge = adjacencyMatrix[u][v];
                HeuristicEdge backEdge = adjacencyMatrix[v][u];
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
                    //frontEdge.cost = edgeCost;
                    frontEdge.cost = Math.max(edgeCost, 0); //NEED TO THINK ABOUT THIS!
                    adjacencyCosts[u][v] = Math.max(edgeCost, 0);
                }
            }
        }
    }

    // For a given src/snk pair, set the cost of the edgs to carry transferAmount of CO2
    public void setRemovalGraphCosts(Source src, Sink snk, double transferAmount) {
        
        for (int u = 0; u < graphVertices.length; u++) {
            for (int v = 0; v < graphVertices.length; v++) {
                HeuristicEdge frontEdge = adjacencyMatrix[u][v];
                HeuristicEdge backEdge = adjacencyMatrix[v][u];
                double edgeCost = 0;

                if(frontEdge!=null) {
                    if(frontEdge.currentHostingAmount>=transferAmount) {
                        edgeCost=1;
                    }

                    frontEdge.cost = Math.max(edgeCost, 0); //NEED TO THINK ABOUT THIS!               
                    adjacencyCosts[u][v] = Math.max(edgeCost, 0);
                }
                

            }
        }
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

    //Run Dijkstras on existing graph edges that are hosting at least amount.
    public Object[] removeDijkstra(Source src,Sink snk,double amount) {
    
        int srcVertexNum = cellNumToVertexNum.get(src.getCellNum());
        int snkVertexNum = cellNumToVertexNum.get(snk.getCellNum());

        int numNodes = graphVertices.length;
        PriorityQueue<ImproveHeuristic.Data> pQueue = new PriorityQueue<>(numNodes);
        double[] costs = new double[numNodes];
        int[] previous = new int[numNodes];
        ImproveHeuristic.Data[] map = new ImproveHeuristic.Data[numNodes];

        for (int vertex = 0; vertex < numNodes; vertex++) {
            costs[vertex] = Double.MAX_VALUE;
            previous[vertex] = -1;
            map[vertex] = new ImproveHeuristic.Data(vertex, costs[vertex]);
        }

        costs[srcVertexNum] = 0;
        map[srcVertexNum].distance = 0;
        pQueue.add(map[srcVertexNum]);
        
        while (!pQueue.isEmpty()) {
            ImproveHeuristic.Data u = pQueue.poll();
            for (int neighbor : neighbors.get(u.vertexNum)) {
                if (adjacencyMatrix[u.vertexNum][neighbor] != null 
                        && adjacencyMatrix[u.vertexNum][neighbor].currentHostingAmount-amount>=-tolerance) {
                    
                    //double altDistance = costs[u.vertexNum] + adjacencyMatrix[u.vertexNum][neighbor].cost;
                    double altDistance = costs[u.vertexNum] + adjacencyCosts[u.vertexNum][neighbor];
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
            if(previousNode==-1) {
                return new Object[]{null,Double.POSITIVE_INFINITY};
            }
            
            path.add(adjacencyMatrix[previousNode][node]);
            node = previousNode;

        }
        

        return new Object[]{path, costs[snkVertexNum]};

    }
    
    // Dijkstra to run on graph edges
    public Object[] dijkstra(Source src, Sink snk) {
        int srcVertexNum = cellNumToVertexNum.get(src.getCellNum());
        int snkVertexNum = cellNumToVertexNum.get(snk.getCellNum());

        int numNodes = graphVertices.length;
        PriorityQueue<ImproveHeuristic.Data> pQueue = new PriorityQueue<>(numNodes);
        double[] costs = new double[numNodes];
        int[] previous = new int[numNodes];
        ImproveHeuristic.Data[] map = new ImproveHeuristic.Data[numNodes];

        for (int vertex = 0; vertex < numNodes; vertex++) {
            costs[vertex] = Double.MAX_VALUE;
            previous[vertex] = -1;
            map[vertex] = new ImproveHeuristic.Data(vertex, costs[vertex]);
        }

        costs[srcVertexNum] = 0;
        map[srcVertexNum].distance = 0;
        pQueue.add(map[srcVertexNum]);

        while (!pQueue.isEmpty()) {
            ImproveHeuristic.Data u = pQueue.poll();
            for (int neighbor : neighbors.get(u.vertexNum)) {
                if (adjacencyMatrix[u.vertexNum][neighbor] != null) {
                    //double altDistance = costs[u.vertexNum] + adjacencyMatrix[u.vertexNum][neighbor].cost;
                    double altDistance = costs[u.vertexNum] + adjacencyCosts[u.vertexNum][neighbor];
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
    
    public HashSet<Source> getSolutionSources() {
        return(solutionSources);
    }

    public HashSet<Sink> getSolutionSinks() {
        return(solutionSinks);
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
        public double amount;

        public Pair(Source src, Sink snk, HashSet<HeuristicEdge> path, double cost,double amount) {
            this.src = src;
            this.snk = snk;
            this.path = path;
            this.cost = cost;
            this.amount=amount;
        }
    }
}
