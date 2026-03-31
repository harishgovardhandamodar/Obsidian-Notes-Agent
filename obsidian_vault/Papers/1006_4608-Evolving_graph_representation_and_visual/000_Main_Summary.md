**000_Main_Summary.md**
==========================

**Evolving Graph Optimization Summary**
----------------------------------------

This repository explores the optimization of evolving graphs, which are sequences of graph instances that change over time. The goal is to minimize vertex movements between consecutive graph instances.

**Key Concepts**

* **Evolving Graphs**: Sequences of graph instances with changing vertices and edges.
* **Vertex Optimization**: Minimizing vertex movements between consecutive graph instances.
* **Window Size**: The number of graph instances used to compute vertex placement.

**Algorithms**

* **Vertex Optimization Algorithm**: Uses a window size to compute vertex placement, reducing total distance measurement (tdEG).
* **Vector Optimization Algorithm**: Enhances vertex optimization by penalizing non-aligned vertex movements, further reducing tdEG.

**Experimental Results**

* **Generated Evolving Graphs**:
	+ Random, Exponential, and Scale-Free graphs were tested with varying window sizes.
	+ Larger window sizes decrease total distance (tdEG) but increase computation time.
	+ Optimal window size depends on the trade-off between cost and computation time. [[005_Experimental_Results_Generated_Graphs.md]]
* **Specific Evolving Graphs**:
	+ Five custom evolving graphs were created to demonstrate the algorithms' effects.
	+ Results show reduced tdEG with increased window sizes. [[006_Experimental_Results_Specific_Graphs.md]]
* **Real-World Evolving Graphs**:
	+ Eurovision and US House of Representatives evolving graphs were optimized using the vertex optimization algorithm.
	+ Significant reductions in tdEG were observed. [[007_Experimental_Results_Real_World_Graphs.md]]

**Process Diagram**
```mermaid
graph LR
    A[Input: Evolving Graph] -->|Window Size|> B[Vertex Optimization Algorithm]
    B --> C[Total Distance Measurement (tdEG)]
    C -->|Comparison|> D[Optimal Window Size]
    D --> E[Output: Optimized Evolving Graph]
    style A fill:#f9f,stroke:#333,stroke-width:2px
    style B fill:#bbf,stroke:#333,stroke-width:2px
    style C fill:#ffb,stroke:#333,stroke-width:2px
    style D fill:#bff,stroke:#333,stroke-width:2px
    style E fill:#f9f,stroke:#333,stroke-width:2px
```
**Related Notes**

* [[001_Vertex_Optimization_Algorithm.md]]: Detailed explanation of the Vertex Optimization Algorithm.
* [[002_Vector_Optimization_Algorithm.md]]: Detailed explanation of the Vector Optimization Algorithm.
* [[003_Experimental_Methodology.md]]: Overview of the experimental setup and methodology.
* [[004_Discussion_and_Conclusion.md]]: In-depth discussion and conclusion of the research findings.