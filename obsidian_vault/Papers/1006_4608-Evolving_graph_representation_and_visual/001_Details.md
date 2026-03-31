**001_Details.md**
=================

**Evolving Graph Optimization: Vertex and Vector Approaches**
---------------------------------------------------------

### Table of Contents
1. [Introduction](#introduction)
2. [Generated Evolving Graphs Experiment](#generated-evolving-graphs-experiment)
3. [Custom Evolving Graphs Experiment](#custom-evolving-graphs-experiment)
4. [Real-World Evolving Graphs: Eurovision and US House of Representatives](#real-world-evolving-graphs)
5. [Comparison of Vertex and Vector Optimization](#comparison-of-vertex-and-vector-optimization)
6. [Conclusion](#conclusion)

### Introduction
---------------

Evolving graphs are dynamic networks where vertices and edges change over time. Optimizing the layout of these graphs is crucial for clear visualization and analysis. This document explores two optimization approaches: **Vertex Optimization** and **Vector Optimization**, applied to various evolving graph scenarios.

### Generated Evolving Graphs Experiment
--------------------------------------

#### Methodology

*   Created 3 types of evolving graphs with different degree distributions:
    *   **Random Evolving Graphs**: Poisson degree distribution
    *   **Exponential Evolving Graphs**: Exponential degree distribution
    *   **Scale-Free Evolving Graphs**: Power-law degree distribution
*   Each type has 20 graph instances with increasing complexity
*   Applied **Vertex Optimization** with varying window sizes (1 to 10)
*   Measured:
    *   **Total Distance (tdEG)**: Sum of all vertex movements across instances
    *   **Computation Time**

#### Results

```mermaid
graph LR
    A[Window Size] -->|Increases|> B[Total Distance (tdEG)]
    B -->|Decreases|
    A -->|Increases|> C[Computation Time]
    C -->|Increases|
```

*   **Total Distance (tdEG)** decreases as window size increases for all graph types
*   Random Evolving Graphs have higher tdEG values due to more vertices
*   Exponential and Scale-Free graphs show similar tdEG patterns
*   Computation time increases with larger window sizes

### Custom Evolving Graphs Experiment
--------------------------------------

#### Methodology

*   Designed 5 custom evolving graphs to test specific scenarios:
    *   **Evolving Graph 1**: Static line graph
    *   **Evolving Graph 2**: Alternating between line graph and empty graph
    *   **Evolving Graph 3**: Transition from line graph to a complex network
*   Applied **Vertex Optimization** with optimal window size (determined from previous experiment)
*   Observed vertex movement patterns

#### Results

| Evolving Graph | Description | Vertex Movement |
| --- | --- | --- |
| 1 | Static Line Graph | No movement |
| 2 | Alternating Structures | Minimal movement |
| 3 | Increasing Complexity | Gradual movement increase |

### Real-World Evolving Graphs
---------------------------

#### Eurovision Evolving Graph

*   Represents voting patterns over years
*   Applied **Vertex Optimization** to visualize pattern changes

#### US House of Representatives Evolving Graph

*   Models member interactions and party dynamics
*   Utilized **Vertex Optimization** for clearer dynamic representation

### Comparison of Vertex and Vector Optimization
----------------------------------------------

#### Methodology

*   Selected a representative evolving graph from each experiment type
*   Applied both **Vertex Optimization** and **Vector Optimization**
*   Compared:
    *   **Total Distance (tdEG)**
    *   **Computation Time**
    *   **Visual Clarity**

#### Results

| Optimization | Total Distance (tdEG) | Computation Time | Visual Clarity |
| --- | --- | --- | --- |
| Vertex | Lower for small window sizes | Generally faster | Good for simple graphs |
| Vector | Better for large window sizes or complex graphs | Can be slower | Excellent for dynamic patterns |

### Conclusion
----------

*   **Vertex Optimization** is suitable for evolving graphs with simple, gradual changes, offering a good balance between total distance and computation time.
*   **Vector Optimization** excels in scenarios with complex dynamics or when visual clarity of pattern changes is paramount, despite potentially higher computation times.
*   The choice of optimization technique depends on the specific characteristics of the evolving graph and the priorities of the analysis (cost vs. visualization quality).

### Related Notes

*   [[002_VertexOptimizationDetails.md]]: In-depth explanation of Vertex Optimization algorithm
*   [[003_VectorOptimizationExplained.md]]: Detailed overview of Vector Optimization methodology
*   [[004_EvolvingGraphsInPractice.md]]: Applications and case studies of evolving graphs in various fields