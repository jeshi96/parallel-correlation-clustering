# Shared-memory Parallel Clustering

Organization
--------

This repository contains the implementations for our
shared-memory parallel and sequential algorithms for correlation and modularity 
clustering.  Note
that the repository uses the
[Graph-Based Benchmark Suite (GBBS)](https://github.com/ParAlg/gbbs)
for parallel primitives and benchmarks.

`clustering` contains the main executable, `cluster-in-memory_main`.
It serves as a central runner for all of our implementations.

<!---
`scripts/test_approx_kcore.py` is a script to run tests for ParallelLDS, 
LDS, and KCore.
-->

## Installation

Compiler:
* g++ &gt;= 7.4.0 with support for Cilk Plus
* g++ &gt;= 7.4.0 with pthread support for the [ParlayLib](https://github.com/cmuparlay/parlaylib) scheduler


Build system:
* [Bazel](https:://bazel.build) &gt;= 3.5.0 and &lt; 4.0.0

To build:
```sh
$ bazel build //clustering:cluster-in-memory_main
```

Most optionality from the [Graph Based Benchmark Suite (GBBS)](https://github.com/ParAlg/gbbs)
applies. In particular, to compile benchmarks for graphs with
more than 2^32 edges, the `LONG` command-line parameter should be set.

Also, the parallel scheduler can be chosen using comand-line parameters.
The default compilation is serial, and the parameters HOMEGROWN, CILK, and
OPENMP switch between the Homegrown scheduler, Cilk Plus, and OpenMP
respectively.

### Graph Format

The applications take as input the adjacency graph format used by
[Graph Based Benchmark Suite (GBBS)](https://github.com/ParAlg/gbbs).

### Configurations

All clusterers take configs given
in a protobuf, as detailed in config.proto, which can be passed in by setting
the input clusterer_config. Note that the proto should be passed in text format.
For both correlation and modularity clustering, a `CorrelationClustererConfig`
should be passed in.

The options available in `CorrelationClustererConfig` are:
* `resolution`: double indicating resolution parameter
* `louvain_config`: `LouvainConfig` object, with
    *  `num_iterations`: int specifying the number of iterations of best moves and compression steps
    * `num_inner_iterations`: int specifying the number of iterations of local best move steps
* `refine`: bool indicating whether to perform multi-level refinement
* `async`: bool indicating whether to use the asynchronous setting
* `move_method`: `MoveMethod` enum, that sets the subset of vertices to consider for best moves to be:
    * `NBHR_CLUSTER_MOVE`: neighbors of clusters that have been modified
    * `ALL_MOVE`: all vertices
    * `NBHR_MOVE`: neighbors of vertices that have moved
* `all_iter`: bool solely for the sequential implementations; indicates if the restriction on the number of iterations should be ignored and if the program should run to convergence

## Example Usage

The main executable is `cluster-in-memory_main` in the `clustering` directory. It
will run the clusterer given by the input clusterer_name: "ParallelCorrelationClusterer",
"ParallelModularityClusterer", "CorrelationClusterer", or "ModularityClusterer".

A template command is:

```
bazel run -c opt  //clustering:cluster-in-memory_main -- --input_graph=</path/to/graph> --output_clustering=</path/to/output> --clusterer_name=<clusterer name> --clusterer_config='<clusterer proto>'
```
