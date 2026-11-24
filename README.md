# Python hardware architecture interview practice

## Run Python

This folder uses Python 3.14 and an isolated `.venv` environment. No third-party packages are required. Python programs can run directly without a separate build step.

From this folder:

```sh
source .venv/bin/activate
python hello.py
python hash.py
```

`hash.py` is currently empty, so running it produces no output until you add code. You can also run any file without activating the environment:

```sh
.venv/bin/python hello.py
```

To recreate the environment, use `python3 -m venv .venv`. If you need a package later, activate the environment and use `python -m pip install PACKAGE_NAME`.

## VS Code

Install the recommended Python and Python Debugger extensions if prompted. Use **Python: Select Interpreter** to select `.venv/bin/python` if VS Code already has a different interpreter selected.

Open a Python file and use **Run Python File in Terminal**, or press **F5** to debug with breakpoints. The **Tasks: Run Task → Python: Run Current File** task also runs the active file through this environment.

## Kernel interview questions

Here, a kernel means a small compute routine. Start with correct pure Python, then explain how you would map the computation to a CPU, GPU, or accelerator. Python lists and interpreter overhead do not model contiguous numeric arrays accurately: use the specified data layouts for hardware analysis, and do not infer hardware throughput from pure-Python timing.

### 1. Vector addition and the roofline model — warm-up

Implement `vector_add(a, b)` for equal-length vectors. Handle empty inputs and reject unequal lengths. A reference example is in `hello.py`.

Assume contiguous FP32 arrays, inputs fetched once from DRAM, and one output store with no write-allocate traffic. How many bytes and floating-point operations are needed per element? At 100 GB/s sustained memory bandwidth, what is the ideal elements/second bound? Why might increasing compute throughput fail to improve performance? How would SIMD and cache-line alignment affect the implementation?

### 2. Matrix multiplication and tiling — medium

Implement `matmul(A, B)` for an M×K matrix and a K×N matrix, then implement a tiled version with a configurable tile size. Support dimensions that are not multiples of the tile size. Check both versions on a small hand-computed example.

Assume contiguous row-major FP32 storage. Compare `i-j-k` and `i-k-j` loop orders. Which accesses have spatial locality? Estimate storage for one tile each of A, B, and C, and choose a tile size for a 32 KiB cache. Explain how reuse changes arithmetic intensity and why the largest possible tile may not be fastest.

### 3. Parallel reduction — medium

Implement a sequential sum and a pairwise tree reduction. Handle empty inputs and odd lengths. Trace the reduction of `[1, 2, 3, 4, 5]` by hand.

Compare total work and dependency depth. How would you partition the reduction across processing elements? Where is synchronization needed? Why can floating-point results change with reduction order? Explain the tradeoff between a shared atomic accumulator and local partial sums.

### 4. Matrix transpose and memory access — medium

Implement a rectangular matrix transpose and a blocked transpose. Verify that transposing twice reconstructs the input, including matrices with dimensions smaller than a block.

For contiguous row-major storage, explain why a simple loop makes either reads or writes strided. How does blocking help cache locality? For a GPU implementation, explain coalesced accesses and how shared-memory tile padding can reduce bank conflicts.

### 5. A 2D stencil and streaming hardware — medium/hard

Implement a five-point stencil: each interior output is the average of the input center and its four direct neighbors. Preserve boundary values and write results to a separate output grid. Specify behavior for grids smaller than 3×3.

Explain why updating the input in place changes the computation. For a raster-streaming FPGA implementation, describe the line buffers and horizontal shift registers you need. Under what input-bandwidth and arithmetic-pipeline assumptions could you produce one interior result per cycle? Distinguish latency from initiation interval, and explain how boundaries affect utilization.

For each exercise, practice explaining correctness, edge cases, operation count, memory traffic, bottlenecks, and what changes when the working set exceeds cache capacity.
