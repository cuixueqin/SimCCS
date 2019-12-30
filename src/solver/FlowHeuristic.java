/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package solver;

import dataStore.DataInOut;
import dataStore.DataStorer;
import dataStore.Edge;
import dataStore.LinearComponent;
import dataStore.Sink;
import dataStore.Solution;
import dataStore.Source;
import dataStore.UnidirEdge;
import ilog.concert.*;
import ilog.cplex.*;
import java.io.File;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 *
 * @author yaw
 */
public class FlowHeuristic {

    private static double bestCost = Double.MAX_VALUE;
    private static double bestPhaseCost = Double.MAX_VALUE;
    private static HashMap<Edge, int[]> numIterations = new HashMap<>();
    private static HashMap<Edge, double[]> sumFlow = new HashMap<>();
    private static HashMap<Edge, double[]> maxFlow = new HashMap<>();
    private static HashMap<Edge, double[]> pastRho = new HashMap<>();
    private static HashMap<Edge, boolean[]> bestOpenedEdges;

    public static void run(DataStorer data, double crf, double numYears, double captureTarget, String basePath, String dataset, String scenario) {
        // create directory
        HashMap<Edge, double[]> edgeHostingAmounts = new HashMap<>();
        DateFormat dateFormat = new SimpleDateFormat("ddMMyyy-HHmmssss");
        Date date = new Date();
        String run = "flowCap" + dateFormat.format(date);
        File solutionDirectory = new File(basePath + "/" + dataset + "/Scenarios/" + scenario + "/Results/" + run + "/");
        solutionDirectory.mkdir();

        int iterationNum = 0;
        double bestSolution = Double.MAX_VALUE;
        double phaseSolution = Double.MAX_VALUE;
        double lastSolution = Double.MAX_VALUE;
        int stagnantIterationMax = 5;
        int worseIterationMaxIntensify = 20;
        int worseIterationMaxDiversify = 5;
        int numUnchanged = 0;
        int numWorse = 0;
        int phase = 0;
        while (iterationNum < 1000) {
            double value = Double.MAX_VALUE;
            if (phase == 0) {
                System.out.print("Seed Iteration " + iterationNum + "(" + bestSolution + "): ");
                value = makeSolveLP2(data, crf, numYears, captureTarget, solutionDirectory.toString(), iterationNum++, phase, false);
            } else if (phase == 1) {
                System.out.print("Intensify Iteration " + iterationNum + "(" + bestSolution + "): ");
                value = makeSolveLP2(data, crf, numYears, captureTarget, solutionDirectory.toString(), iterationNum++, phase, false);
            } else {
                System.out.print("Diversify Iteration " + iterationNum + "(" + bestSolution + "): ");
                value = makeSolveLP2(data, crf, numYears, captureTarget, solutionDirectory.toString(), iterationNum++, phase, false);
            }

            if (value == lastSolution) {
                numUnchanged++;
                numWorse = 0;
            } else if (value > phaseSolution) {
                numWorse++;
                numUnchanged = 0;
            } else {
                numUnchanged = 0;
                numWorse = 0;
            }
            
            if (value < phaseSolution) {
                phaseSolution = value;
            }

            lastSolution = value;

            // When to end phase
            if (numUnchanged >= stagnantIterationMax || (numWorse >= worseIterationMaxIntensify && phase == 0) || (numWorse >= worseIterationMaxIntensify && phase == 1) || (numWorse >= worseIterationMaxDiversify && phase == 2)) {
                if (phase == 0) {
                    phase = 1;
                } else if (phase == 1) {
                    phase = 2;
                } else {
                    phase = 1;
                }
                
                System.out.print("Local Improvement: ");
                value = makeSolveLP2(data, crf, numYears, captureTarget, solutionDirectory.toString(), iterationNum, 1, true);

                if (value < phaseSolution) {
                    phaseSolution = value;
                }

                if (phaseSolution < bestSolution) {
                    bestSolution = phaseSolution;
                }

                lastSolution = Double.MAX_VALUE;
                phaseSolution = Double.MAX_VALUE;
                numUnchanged = 0;
                numWorse = 0;
                bestPhaseCost = Double.MAX_VALUE;
            }
        }
    }

    private static double makeSolveLP2(DataStorer data, double crf, double numYears, double captureTarget, String solutionDirectory, int iterationNum, int phase, boolean localImprovement) {
        double solutionCost = 0;

        Source[] sources = data.getSources();
        Sink[] sinks = data.getSinks();
        LinearComponent[] linearComponents = data.getLinearComponents();
        int[] graphVertices = data.getGraphVertices();
        HashMap<Integer, HashSet<Integer>> neighbors = data.getGraphNeighbors();
        HashMap<Edge, Double> edgeConstructionCosts = data.getGraphEdgeConstructionCosts();

        //HashMap<Edge, Double> edgeRightOfWayCosts = data.getGraphEdgeRightOfWayCosts();
        HashMap<Source, Integer> sourceCellToIndex = new HashMap<>();
        HashMap<Integer, Source> sourceIndexToCell = new HashMap<>();
        HashMap<Sink, Integer> sinkCellToIndex = new HashMap<>();
        HashMap<Integer, Sink> sinkIndexToCell = new HashMap<>();
        HashMap<Integer, Integer> vertexCellToIndex = new HashMap<>();
        HashMap<Integer, Integer> vertexIndexToCell = new HashMap<>();
        HashMap<UnidirEdge, Integer> edgeToIndex = new HashMap<>();
        HashMap<Integer, UnidirEdge> edgeIndexToEdge = new HashMap<>();
        HashSet<Integer> sourceCells = new HashSet<>();
        HashSet<Integer> sinkCells = new HashSet<>();

        // Initialize cell/index maps
        for (int i = 0; i < sources.length; i++) {
            sourceCellToIndex.put(sources[i], i);
            sourceIndexToCell.put(i, sources[i]);
            sourceCells.add(sources[i].getCellNum());
        }
        for (int i = 0; i < sinks.length; i++) {
            sinkCellToIndex.put(sinks[i], i);
            sinkIndexToCell.put(i, sinks[i]);
            sinkCells.add(sinks[i].getCellNum());
        }
        for (int i = 0; i < graphVertices.length; i++) {
            vertexCellToIndex.put(graphVertices[i], i);
            vertexIndexToCell.put(i, graphVertices[i]);
        }
        int index = 0;
        for (Edge e : edgeConstructionCosts.keySet()) {
            UnidirEdge e1 = new UnidirEdge(e.v1, e.v2);
            edgeToIndex.put(e1, index);
            edgeIndexToEdge.put(index, e1);
            index++;

            UnidirEdge e2 = new UnidirEdge(e.v2, e.v1);
            edgeToIndex.put(e2, index);
            edgeIndexToEdge.put(index, e2);
            index++;
        }

        try {
            IloCplex cplex = new IloCplex();

            // variable: a
            IloNumVar[] a = new IloNumVar[sources.length];
            for (int s = 0; s < sources.length; s++) {
                a[s] = cplex.numVar(0, Double.MAX_VALUE, "a[" + s + "]");
            }

            // variable: b
            IloNumVar[] b = new IloNumVar[sinks.length];
            for (int s = 0; s < sinks.length; s++) {
                b[s] = cplex.numVar(0, Double.MAX_VALUE, "b[" + s + "]");
            }

            // variable: p
            IloNumVar[][] p = new IloNumVar[edgeToIndex.size()][linearComponents.length];
            for (int e = 0; e < edgeToIndex.size(); e++) {
                for (int c = 0; c < linearComponents.length; c++) {
                    p[e][c] = cplex.numVar(0, Double.MAX_VALUE, "p[" + e + "][" + c + "]");
                }
            }

            // constraint A: pipe capacity
            for (int e = 0; e < edgeToIndex.size(); e++) {
                for (int c = 0; c < linearComponents.length; c++) {
                    IloLinearNumExpr expr = cplex.linearNumExpr();
                    expr.addTerm(p[e][c], 1.0);
                    cplex.addLe(expr, linearComponents[c].getMaxCapacity());
                }
            }

            // constraint B: conservation of flow
            for (int src : graphVertices) {
                IloLinearNumExpr expr = cplex.linearNumExpr();
                for (int dest : neighbors.get(src)) {
                    UnidirEdge edge = new UnidirEdge(src, dest);
                    for (int c = 0; c < linearComponents.length; c++) {
                        expr.addTerm(p[edgeToIndex.get(edge)][c], 1.0);
                    }
                }

                for (int dest : neighbors.get(src)) {
                    UnidirEdge edge = new UnidirEdge(dest, src);
                    for (int c = 0; c < linearComponents.length; c++) {
                        expr.addTerm(p[edgeToIndex.get(edge)][c], -1.0);
                    }
                }

                // Set right hand side
                if (sourceCells.contains(src)) {
                    for (Source source : sources) {
                        if (source.getCellNum() == src) {
                            expr.addTerm(a[sourceCellToIndex.get(source)], -1.0);
                        }
                    }
                }
                if (sinkCells.contains(src)) {
                    for (Sink sink : sinks) {
                        if (sink.getCellNum() == src) {
                            expr.addTerm(b[sinkCellToIndex.get(sink)], 1.0);
                        }
                    }
                }
                cplex.addEq(expr, 0.0);
            }

            // constraint C: capture limit
            for (int s = 0; s < sources.length; s++) {
                IloLinearNumExpr expr = cplex.linearNumExpr();
                expr.addTerm(a[s], 1.0);
                cplex.addLe(expr, sources[s].getProductionRate());
            }

            // constraint D: injection limit
            for (int s = 0; s < sinks.length; s++) {
                IloLinearNumExpr expr = cplex.linearNumExpr();
                expr.addTerm(b[s], 1.0);
                cplex.addLe(expr, sinks[s].getCapacity() / numYears);
            }

            // constraint E: capture target
            IloLinearNumExpr expr = cplex.linearNumExpr();
            for (int s = 0; s < sources.length; s++) {
                expr.addTerm(a[s], 1.0);
            }
            cplex.addGe(expr, captureTarget);

            // constraint H: hardcoded values
            IloNumVar captureTargetVar = cplex.numVar(captureTarget, captureTarget, "captureTarget");
            IloLinearNumExpr h1 = cplex.linearNumExpr();
            h1.addTerm(captureTargetVar, 1.0);
            cplex.addEq(h1, captureTarget);

            IloNumVar crfVar = cplex.numVar(crf, crf, "crf");
            IloLinearNumExpr h2 = cplex.linearNumExpr();
            h2.addTerm(crfVar, 1.0);
            cplex.addEq(h2, crf);

            IloNumVar projectLengthVar = cplex.numVar(numYears, numYears, "projectLength");
            IloLinearNumExpr h3 = cplex.linearNumExpr();
            h3.addTerm(projectLengthVar, 1.0);
            cplex.addEq(h3, numYears);

            // objective
            IloLinearNumExpr objExpr = cplex.linearNumExpr();
            for (int s = 0; s < sources.length; s++) {
                objExpr.addTerm(a[s], sources[s].getCaptureCost());
            }

            for (int s = 0; s < sinks.length; s++) {
                objExpr.addTerm(b[s], sinks[s].getInjectionCost());
            }

            // determine average and std deviation of num interations
            double deltaPlus = 0;
            double deltaMinus = 0;
            if (!numIterations.isEmpty()) {
                int sumIterations = 0;
                int numInAvg = 0;
                for (Edge e : numIterations.keySet()) {
                    int[] values = numIterations.get(e);
                    for (int nI : values) {
                        if (nI > 0) {
                            numInAvg++;
                            sumIterations += nI;
                        }
                    }
                }
                double avgIterations = sumIterations / numInAvg;

                double stdDev = 0;
                for (Edge e : numIterations.keySet()) {
                    int[] values = numIterations.get(e);
                    for (int nI : values) {
                        if (nI > 0) {
                            stdDev += Math.pow(nI - avgIterations, 2);
                        }
                    }
                }
                stdDev = Math.sqrt(stdDev / numInAvg);

                deltaPlus = avgIterations + (.5) * stdDev; // "typically calues in [0,1] show good performance
                deltaMinus = avgIterations - (0) * stdDev;
            }

            for (int e = 0; e < edgeToIndex.size(); e++) {
                for (int c = 0; c < linearComponents.length; c++) {
                    UnidirEdge unidirEdge = edgeIndexToEdge.get(e);
                    Edge bidirEdge = new Edge(unidirEdge.v1, unidirEdge.v2);

                    double fixedCost = (linearComponents[c].getConIntercept() * edgeConstructionCosts.get(bidirEdge)) * crf;
                    double variableCost = (linearComponents[c].getConSlope() * edgeConstructionCosts.get(bidirEdge)) * crf;

                    double rho = fixedCost / 1; //linearComponents[c].getMaxCapacity();
                    if (pastRho.containsKey(bidirEdge)) {
                        double[] values = pastRho.get(bidirEdge);
                        if (values[c] > 0) {
                            rho = values[c];
                            double v = (sumFlow.get(bidirEdge)[c] / (iterationNum + 1)) / maxFlow.get(bidirEdge)[c]; //assumes numbering starts at 0

                            if (phase == 1) {
                                //phase = 1 -> Intensify
                                if (numIterations.get(bidirEdge)[c] >= deltaPlus) {
                                    rho *= 1 - v;
                                    //pastRho.get(bidirEdge)[c] = rho;
                                } else if (numIterations.get(bidirEdge)[c] < deltaMinus) {
                                    rho *= 2 - v;
                                    //pastRho.get(bidirEdge)[c] = rho;
                                }
                            } else if (phase == 2) {
                                //phase = 2 -> Diversify
                                if (numIterations.get(bidirEdge)[c] >= deltaPlus) {
                                    rho *= 1 + v;
                                    //pastRho.get(bidirEdge)[c] = rho;
                                } else if (numIterations.get(bidirEdge)[c] < deltaMinus) {
                                    rho *= v;
                                    //pastRho.get(bidirEdge)[c] = rho;
                                }
                            }
                        }
                    }
                    if (localImprovement) {
                        if (bestOpenedEdges.containsKey(bidirEdge)) {
                            if (bestOpenedEdges.get(bidirEdge)[c]) {
                                rho = 0;
                            } else {
                                rho = 1000000;
                            }
                        } else {
                            rho = 1000000;
                        }
                    }
                    objExpr.addTerm(p[e][c], variableCost + rho);
                }
            }

            // objective:
            IloObjective obj = cplex.minimize(objExpr);
            cplex.add(obj);
            cplex.setOut(null);

            // solve
            HashMap<Edge, boolean[]> openedEdges = new HashMap<>();
            if (cplex.solve()) {
                for (int e = 0; e < edgeToIndex.size(); e++) {
                    for (int c = 0; c < linearComponents.length; c++) {
                        if (cplex.getValue(p[e][c]) > 0.0001) {
                            UnidirEdge unidirEdge = edgeIndexToEdge.get(e);
                            Edge bidirEdge = new Edge(unidirEdge.v1, unidirEdge.v2);

                            if (!numIterations.containsKey(bidirEdge)) {
                                numIterations.put(bidirEdge, new int[linearComponents.length]);
                                sumFlow.put(bidirEdge, new double[linearComponents.length]);
                                maxFlow.put(bidirEdge, new double[linearComponents.length]);
                                pastRho.put(bidirEdge, new double[linearComponents.length]);
                            }

                            if (!openedEdges.containsKey(bidirEdge)) {
                                openedEdges.put(bidirEdge, new boolean[linearComponents.length]);
                            }

                            numIterations.get(bidirEdge)[c] += 1;
                            sumFlow.get(bidirEdge)[c] += cplex.getValue(p[e][c]);
                            if (cplex.getValue(p[e][c]) > maxFlow.get(bidirEdge)[c]) {
                                maxFlow.get(bidirEdge)[c] = cplex.getValue(p[e][c]);
                            }

                            double fixedCost = (linearComponents[c].getConIntercept() * edgeConstructionCosts.get(bidirEdge)) * crf;
                            pastRho.get(bidirEdge)[c] = fixedCost / cplex.getValue(p[e][c]);
                            openedEdges.get(bidirEdge)[c] = true;
                        }
                    }
                }
            } else {
                System.out.println("Not Feasible");
            }

            if (solutionDirectory != null && solutionDirectory != "") {
                cplex.exportModel(solutionDirectory + "/flowCap.mps");
                cplex.writeSolution(solutionDirectory + "/solution.sol");
                Solution soln = DataInOut.loadSolution(solutionDirectory, -1);
                double cost = soln.getTotalCost();
                System.out.println("Solution Cost = " + cost);
                if (cost < bestCost) {
                    bestCost = cost;
                    cplex.exportModel(solutionDirectory + "/BESTflowCap.mps");
                    cplex.writeSolution(solutionDirectory + "/BESTsolution.sol");
                }
                if (cost < bestPhaseCost) {
                    bestOpenedEdges = openedEdges;
                }
                solutionCost = cost;
            }
            cplex.clearModel();
        } catch (IloException ex) {
            Logger.getLogger(FlowHeuristic.class.getName()).log(Level.SEVERE, null, ex);
        }
        return solutionCost;
    }
}
