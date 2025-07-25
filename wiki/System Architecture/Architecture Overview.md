<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [deprecated/hw1/hw1.md](https://github.com/agattani123/cis6010/blob/main/deprecated/hw1/hw1.md)
- [deprecated/hw2/hw2.sln](https://github.com/agattani123/cis6010/blob/main/deprecated/hw2/hw2.sln)
- [gemm/README.md](https://github.com/agattani123/cis6010/blob/main/gemm/README.md)
- [gemm/cugemm.cu](https://github.com/agattani123/cis6010/blob/main/gemm/cugemm.cu)
- [gemm/Makefile](https://github.com/agattani123/cis6010/blob/main/gemm/Makefile)

</details>

# Architecture Overview

## Introduction

This wiki page provides an overview of the architecture and implementation details for the General Matrix Multiplication (GEMM) optimization project. The project aims to optimize a CUDA implementation of GEMM, which is a fundamental operation in linear algebra and widely used in various scientific and computational applications.

The project consists of several stages, starting with a naive implementation (`runBasic`) and progressively optimizing it by addressing uncoalesced memory accesses (`runGmemCoalesced`), utilizing shared memory (`runSharedMem`), and employing techniques to compute multiple output elements per thread (`runMultipleResultsPerThread`). The ultimate goal is to achieve high performance and efficiency on the GPU by leveraging various optimization techniques.

Sources: [gemm/README.md](https://github.com/agattani123/cis6010/blob/main/gemm/README.md)

## Project Structure

The project is organized into several files and directories:

- `cugemm.cu`: The main source file containing the CUDA kernels and host code for GEMM implementation.
- `Makefile`: A Makefile for building the project with different optimization levels and profiling configurations.
- `README.md`: A README file providing an overview of the project, build instructions, and optimization tasks.

Sources: [gemm/cugemm.cu](https://github.com/agattani123/cis6010/blob/main/gemm/cugemm.cu), [gemm/Makefile](https://github.com/agattani123/cis6010/blob/main/gemm/Makefile), [gemm/README.md](https://github.com/agattani123/cis6010/blob/main/gemm/README.md)

## GEMM Implementation

The GEMM operation is defined as follows:

```
C = alpha * op(A) * op(B) + beta * C
```

Where `A`, `B`, and `C` are matrices, `alpha` and `beta` are scalar values, and `op(X)` represents either `X` or its transpose, depending on the operation.

The project provides several GEMM implementations, each with different optimization levels:

### runBasic

The `runBasic` function implements a naive GEMM algorithm without any optimizations. It serves as a baseline for performance comparison and profiling.

```mermaid
graph TD
    A[Allocate input and output matrices] --> B[Initialize input matrices with random values]
    B --> C[Run naive GEMM kernel]
    C --> D[Validate GEMM result (optional)]
    D --> E[Deallocate matrices]
```

Sources: [gemm/cugemm.cu:173-208](https://github.com/agattani123/cis6010/blob/main/gemm/cugemm.cu#L173-L208)

### runCublas

The `runCublas` function utilizes the highly optimized cuBLAS library from NVIDIA to perform GEMM operations. It serves as a reference implementation for validation purposes.

```mermaid
graph TD
    A[Allocate input and output matrices] --> B[Initialize input matrices with random values]
    B --> C[Run cuBLAS GEMM kernel]
    C --> D[Validate GEMM result (optional)]
    D --> E[Deallocate matrices]
```

Sources: [gemm/cugemm.cu:210-240](https://github.com/agattani123/cis6010/blob/main/gemm/cugemm.cu#L210-L240)

### runGmemCoalesced

The `runGmemCoalesced` function is an optimized version of `runBasic` that addresses uncoalesced global memory accesses. It aims to improve performance by coalescing global memory accesses, resulting in fewer memory transactions.

```mermaid
graph TD
    A[Allocate input and output matrices] --> B[Initialize input matrices with random values]
    B --> C[Run coalesced GEMM kernel]
    C --> D[Validate GEMM result (optional)]
    D --> E[Deallocate matrices]
```

Sources: [gemm/README.md:46-48](https://github.com/agattani123/cis6010/blob/main/gemm/README.md#L46-L48)

### runSharedMem

The `runSharedMem` function further optimizes the GEMM implementation by utilizing shared memory to cache tiles of the input matrices. This optimization reduces redundant global memory accesses and improves performance.

```mermaid
graph TD
    A[Allocate input and output matrices] --> B[Initialize input matrices with random values]
    B --> C[Run shared memory GEMM kernel]
    C --> D[Validate GEMM result (optional)]
    D --> E[Deallocate matrices]
```

Sources: [gemm/README.md:51-53](https://github.com/agattani123/cis6010/blob/main/gemm/README.md#L51-L53)

### runMultipleResultsPerThread

The `runMultipleResultsPerThread` function employs a technique where each thread computes multiple cells of the output matrix `C`. This optimization improves arithmetic intensity and further enhances performance.

```mermaid
graph TD
    A[Allocate input and output matrices] --> B[Initialize input matrices with random values]
    B --> C[Run multiple results per thread GEMM kernel]
    C --> D[Validate GEMM result (optional)]
    D --> E[Deallocate matrices]
```

Sources: [gemm/README.md:56-58](https://github.com/agattani123/cis6010/blob/main/gemm/README.md#L56-L58)

## Performance Profiling and Debugging

The project provides tools and techniques for performance profiling and debugging:

### Performance Profiling

The Nvidia Compute Insight profiler is used to analyze the performance of the GEMM kernels and identify potential bottlenecks. The profiling data can be collected and viewed using the following commands:

```
sudo /usr/local/cuda-11.8/bin/ncu -o profile-basic --set full ./cugemm-profile.bin --size=4096 --reps=1 --algo=1 --validate=false
```

The profiling results can be viewed using the Nvidia Compute Insight profiler on a local machine.

Sources: [gemm/README.md:28-33](https://github.com/agattani123/cis6010/blob/main/gemm/README.md#L28-L33)

### Debugging

The project leverages Nvidia's compute sanitizers to detect memory safety and concurrency errors in the CUDA kernels. The sanitizers can be run on the debug binaries using the following commands:

```
compute-sanitizer --tool memcheck ./cugemm-debug.bin ...
compute-sanitizer --tool racecheck ./cugemm-debug.bin ...
```

These sanitizers provide valuable information about potential issues in the source code, helping to identify and fix bugs.

Sources: [gemm/README.md:37-40](https://github.com/agattani123/cis6010/blob/main/gemm/README.md#L37-L40)

## Build and Execution

The project can be built and executed using the provided `Makefile`. The following commands can be used:

```
make -j3 all
./cugemm.bin --size=2048 --reps=1 --algo=1
```

This command builds three versions of the code (optimized, optimized with profiling, and unoptimized with debugging symbols) and runs the optimized version with the specified parameters.

Sources: [gemm/README.md:16-20](https://github.com/agattani123/cis6010/blob/main/gemm/README.md#L16-L20), [gemm/Makefile](https://github.com/agattani123/cis6010/blob/main/gemm/Makefile)

## Conclusion

The GEMM optimization project provides a comprehensive framework for optimizing a CUDA implementation of the General Matrix Multiplication operation. It starts with a naive implementation and progressively applies various optimization techniques, such as coalesced memory accesses, shared memory utilization, and computing multiple output elements per thread. The project also includes tools and techniques for performance profiling and debugging, ensuring efficient and correct execution on the GPU.

By following the provided guidelines and leveraging the optimization stages, developers can achieve significant performance improvements and gain insights into optimizing CUDA kernels for matrix operations and other computational workloads.