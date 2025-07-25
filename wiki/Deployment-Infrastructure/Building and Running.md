<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [gemm/Makefile](https://github.com/agattani123/cis6010/blob/main/gemm/Makefile)
- [gemm/launch.json](https://github.com/agattani123/cis6010/blob/main/gemm/launch.json)

</details>

# Building and Running

## Introduction

This wiki page covers the process of building and running the GEMM (General Matrix Multiplication) project, which appears to be a CUDA C++ application for performing matrix multiplication on GPUs. The project provides different build configurations for optimized, debug, and profiling purposes, as well as command-line arguments to control the matrix multiplication algorithm and input size.

## Building the Project

The project uses a Makefile to build different versions of the `cugemm` binary. The following build targets are available:

### `all` (Default)

```makefile
all: cugemm.bin cugemm-debug.bin cugemm-profile.bin
```

This target builds three versions of the `cugemm` binary:

1. `cugemm.bin`: An optimized binary for production use.
2. `cugemm-debug.bin`: A debug binary without optimizations, suitable for debugging.
3. `cugemm-profile.bin`: An optimized binary with line number information for profiling.

Sources: [gemm/Makefile](https://github.com/agattani123/cis6010/blob/main/gemm/Makefile)

### `cugemm.bin`

```makefile
cugemm.bin: $(SOURCE_FILE)
    nvcc -std=c++17 --generate-code=arch=compute_75,code=[compute_75,sm_75] $^ -lcublas -o $@
```

This target compiles the `cugemm.cu` source file using the NVIDIA CUDA compiler (`nvcc`) with the following options:

- `-std=c++17`: Enables C++17 language support.
- `--generate-code=arch=compute_75,code=[compute_75,sm_75]`: Generates code for the NVIDIA Turing architecture (compute capability 7.5).
- `-lcublas`: Links against the CUDA BLAS library for matrix operations.
- `-o $@`: Specifies the output file name (`cugemm.bin`).

Sources: [gemm/Makefile:5-6](https://github.com/agattani123/cis6010/blob/main/gemm/Makefile#L5-L6)

### `cugemm-debug.bin`

```makefile
cugemm-debug.bin: $(SOURCE_FILE)
    nvcc -g -G -src-in-ptx -std=c++17 --generate-code=arch=compute_75,code=[compute_75,sm_75] $^ -lcublas -o $@
```

This target compiles the `cugemm.cu` source file with the following additional options for debugging:

- `-g`: Generates debug information.
- `-G`: Generates relocatable device code.
- `-src-in-ptx`: Includes source code in PTX (Parallel Thread Execution) output.

Sources: [gemm/Makefile:9-10](https://github.com/agattani123/cis6010/blob/main/gemm/Makefile#L9-L10)

### `cugemm-profile.bin`

```makefile
cugemm-profile.bin: $(SOURCE_FILE)
    nvcc -g --generate-line-info -src-in-ptx -std=c++17 --generate-code=arch=compute_75,code=[compute_75,sm_75] $^ -lcublas -o $@
```

This target compiles the `cugemm.cu` source file with the following additional options for profiling:

- `-g`: Generates debug information.
- `--generate-line-info`: Generates line number information for profiling.
- `-src-in-ptx`: Includes source code in PTX (Parallel Thread Execution) output.

Sources: [gemm/Makefile:13-14](https://github.com/agattani123/cis6010/blob/main/gemm/Makefile#L13-L14)

### `profile`

```makefile
profile: cugemm-profile.bin
    sudo /usr/local/cuda-11.8/bin/ncu --export my-profile --set full ./cugemm-profile.bin --size=1024 --reps=1 --algo=1 --validate=false
```

This target runs the NVIDIA Command Line Profiler (`ncu`) on the `cugemm-profile.bin` binary with the following options:

- `--export my-profile`: Exports the profiling data to a file named `my-profile`.
- `--set full`: Sets the profiling mode to "full".
- `--size=1024`: Sets the matrix size to 1024x1024.
- `--reps=1`: Runs the matrix multiplication once.
- `--algo=1`: Selects algorithm 1 for matrix multiplication.
- `--validate=false`: Disables validation of the matrix multiplication result.

Note: The `--algo` flag can be changed to profile a different matrix multiplication algorithm.

Sources: [gemm/Makefile:18-20](https://github.com/agattani123/cis6010/blob/main/gemm/Makefile#L18-L20)

### `clean`

```makefile
clean:
    rm -f cugemm*.bin
```

This target removes all compiled `cugemm` binaries.

Sources: [gemm/Makefile:23-24](https://github.com/agattani123/cis6010/blob/main/gemm/Makefile#L23-L24)

## Running the Project

The `launch.json` file in the project provides configurations for launching and debugging the `cugemm` binary using the Visual Studio Code CUDA debugger extension.

### `CUDA C++: Launch`

```json
{
    "name": "CUDA C++: Launch",
    "type": "cuda-gdb",
    "request": "launch",
    "program": "/home/ubuntu/cis6010/gemm/cugemm-debug.bin",
    "args": "--algo 4 --size 4"
}
```

This configuration launches the `cugemm-debug.bin` binary with the following command-line arguments:

- `--algo 4`: Selects algorithm 4 for matrix multiplication.
- `--size 4`: Sets the matrix size to 4x4.

Sources: [gemm/launch.json:10-15](https://github.com/agattani123/cis6010/blob/main/gemm/launch.json#L10-L15)

### `CUDA C++: Attach`

```json
{
    "name": "CUDA C++: Attach",
    "type": "cuda-gdb",
    "request": "attach"
}
```

This configuration allows attaching the CUDA debugger to a running CUDA process.

Sources: [gemm/launch.json:17-20](https://github.com/agattani123/cis6010/blob/main/gemm/launch.json#L17-L20)

## Conclusion

The GEMM project provides a Makefile for building different versions of the `cugemm` binary, including optimized, debug, and profiling builds. The `launch.json` file allows for easy launching and debugging of the binary using the Visual Studio Code CUDA debugger extension. Command-line arguments can be used to control the matrix multiplication algorithm and input size.