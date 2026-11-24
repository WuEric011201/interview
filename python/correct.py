"""Reference answers to all five kernel exercises in README.md.

Run with: .venv/bin/python python/correct.py
"""


def vector_add(a: list[float], b: list[float]) -> list[float]:
    """ O(n) time and O(n) output storage.

    Each element costs one addition and 12 bytes: two 4-byte loads and one
    4-byte store (without write allocation). Arithmetic intensity is 1/12
    FLOP/byte. At 100 GB/s the bandwidth bound is 100e9/12, about 8.33 billion
    elements/s, also 8.33 GFLOP/s. Extra compute throughput cannot overcome
    this bandwidth limit. SIMD processes multiple elements per instruction;
    alignment can avoid split cache-line accesses, with a scalar/masked tail.
    Cache-resident inputs can use higher cache bandwidth; inputs larger than
    cache must stream from DRAM under the stated traffic assumptions.
    """
    if len(a) != len(b):
        raise ValueError("Vectors must have the same length")
    return [x + y for x, y in zip(a, b)]


def _shape(matrix: list[list[float]]) -> tuple[int, int]:
    """Validate the matrix representation and return (rows, columns)."""
    if not matrix:
        return 0, 0
    columns = len(matrix[0])
    if columns == 0 or any(len(row) != columns for row in matrix):
        raise ValueError("Matrices must be rectangular with nonempty rows")
    return len(matrix), columns


def _check_tile_size(tile_size: int) -> None:
    if isinstance(tile_size, bool) or not isinstance(tile_size, int) or tile_size <= 0:
        raise ValueError("Tile size must be a positive integer")


def matmul(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    """Multiply MxK by KxN using i-j-k order; O(MKN) time, O(MN) output.

    There are MKN multiplications and MKN accumulator additions (~2MKN
    FLOPs). In i-j-k order, A's row is contiguous but B's column is strided.
    In i-k-j order, B and C are contiguous in the inner loop and A[i][k]
    is reused. Large strided working sets cause cache misses and repeated
    DRAM traffic; tiling keeps a smaller working set available for reuse.
    """
    m, k = _shape(A)
    b_rows, n = _shape(B)
    if k != b_rows:
        raise ValueError("A's column count must equal B's row count")
    C = [[0.0] * n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            for p in range(k):
                C[i][j] += A[i][p] * B[p][j]
    return C


def matmul_tiled(
    A: list[list[float]], B: list[list[float]], tile_size: int = 32
) -> list[list[float]]:
    """Multiply with bounded tiles and i-k-j order inside each tile.

    Three square FP32 tiles require 3 * 4 * T^2 bytes. A 32 KiB cache fits
    at most T=52 by capacity alone; T=32 uses 12 KiB and leaves headroom.
    A tile product does ~2T^3 FLOPs. Loading A and B and reading/writing C
    costs ~16T^2 bytes, giving ~T/8 FLOP/byte for that tile update. Keeping
    C resident across updates improves traffic further. Ideally the whole
    multiply transfers 4(MK+KN+MN) bytes, but capacity limits can force
    operand reloads. Cache associativity, other data, SIMD widths, register
    pressure, and loop overhead mean the largest tile need not be fastest.
    Time is O(MKN); output is O(MN), with no explicit tile copies.
    """
    _check_tile_size(tile_size)
    m, k = _shape(A)
    b_rows, n = _shape(B)
    if k != b_rows:
        raise ValueError("A's column count must equal B's row count")
    C = [[0.0] * n for _ in range(m)]
    for ii in range(0, m, tile_size):
        for jj in range(0, n, tile_size):
            for kk in range(0, k, tile_size):
                for i in range(ii, min(ii + tile_size, m)):
                    for p in range(kk, min(kk + tile_size, k)):
                        a_value = A[i][p]
                        for j in range(jj, min(jj + tile_size, n)):
                            C[i][j] += a_value * B[p][j]
    return C


def sequential_sum(values: list[float]) -> float:
    """Sum in order: O(n) work, O(n) dependency depth, O(1) extra space.

    An empty input sums to zero. Starting with the first element uses n-1
    additions for nonempty inputs and reads each input once (~4n FP32 bytes).
    A serial accumulator can be latency-bound; multiple accumulators enable
    parallelism. For large arrays, memory bandwidth can become the limit.
    """
    if not values:
        return 0.0
    total = values[0]
    for i in range(1, len(values)):
        total += values[i]
    return total


def tree_sum(values: list[float]) -> float:
    """Pairwise reduction: O(n) work, O(log n) dependency depth, O(n) space.

    [1,2,3,4,5] -> [3,7,5] -> [10,5] -> [15]. Carry an odd final item
    unchanged. There are n-1 additions for n>0. This Python implementation
    executes serially; logarithmic depth describes potential parallelism.
    Intermediate lists add O(n) traffic beyond the input read.

    Partition the input among P processing elements, reduce locally, then
    combine partial sums in a tree. Synchronize before consuming a previous
    stage's results (or use supported warp/group reduction primitives).
    Floating-point addition is not associative because of rounding, so
    different trees can produce different answers. One shared atomic sum
    serializes contending updates and can have nondeterministic ordering;
    local partial sums reduce contention and communication. For data beyond
    cache, local streaming reductions avoid repeated full-array passes.
    """
    current = list(values)
    while len(current) > 1:
        next_level = [current[i] + current[i + 1]
                      for i in range(0, len(current) - 1, 2)]
        if len(current) % 2:
            next_level.append(current[-1])
        current = next_level
    return current[0] if current else 0.0


def transpose(matrix: list[list[float]]) -> list[list[float]]:
    """Transpose MxN to NxM; O(MN) time and output space, no arithmetic.

    Reading input rows contiguously writes output columns with a stride;
    reversing the loops makes reads strided instead. Ideal traffic is 8MN
    bytes (one FP32 load/store), excluding write allocation. Poor locality
    can transfer many partly used cache lines, especially beyond cache.
    """
    rows, columns = _shape(matrix)
    result = [[0.0] * rows for _ in range(columns)]
    for i in range(rows):
        for j in range(columns):
            result[j][i] = matrix[i][j]
    return result


def transpose_blocked(
    matrix: list[list[float]], block_size: int = 32
) -> list[list[float]]:
    """Transpose tiles, handling partial edge tiles; O(MN) time/output.

    Blocking keeps input/output cache lines live until their nearby elements
    are used. Two FP32 tiles occupy roughly 8T^2 bytes. The kernel remains
    bandwidth-bound, but makes better use of each cache-line transfer.
    On a GPU, neighboring threads should read neighboring input elements
    (coalesced loads), synchronize after staging a shared-memory tile, then
    read it transposed and write neighboring output elements (coalesced
    stores). With 32 banks and one FP32 word per bank, a [32][33] shared
    tile avoids the same-bank column accesses of a [32][32] tile. Padding
    depends on the bank layout and element size. Guard partial edge tiles
    while ensuring all required threads still reach the synchronization.
    """
    _check_tile_size(block_size)
    rows, columns = _shape(matrix)
    result = [[0.0] * rows for _ in range(columns)]
    for ii in range(0, rows, block_size):
        for jj in range(0, columns, block_size):
            for i in range(ii, min(ii + block_size, rows)):
                for j in range(jj, min(jj + block_size, columns)):
                    result[j][i] = matrix[i][j]
    return result


def five_point_stencil(grid: list[list[float]]) -> list[list[float]]:
    """Average each interior cell with its four direct neighbors.

    Boundaries are copied unchanged. Grids with fewer than three rows or
    columns are copied unchanged, including []. Input rows are never mutated
    or shared with output. In-place updates would mix old/new neighbor values
    and make the answer depend on traversal order.

    Time/output space are O(rows*columns). Each interior cell uses four adds
    and one division (or reciprocal multiply). Without reuse, five loads
    and one store cost 24 FP32 bytes per interior result. With row reuse,
    ideal large-grid DRAM traffic approaches one input load and one output
    store per cell (8 bytes), excluding write allocation. This reference
    also copies the input before overwriting interior outputs. If the active
    rows exceed cache, blocking can reduce repeated DRAM reads.

    For raster-streaming FPGA hardware, use two width-sized line delays and
    horizontal shift registers on three rows to expose a 3x3 window; only
    its center and four direct neighbors are needed. Delay output until the
    right/bottom neighbors arrive. One interior result per cycle requires
    one new input per cycle, sufficient line-buffer ports, output bandwidth,
    and a fully pipelined arithmetic path accepting a new window each cycle.
    A constant reciprocal multiply may replace division if its rounding is
    acceptable. Initiation interval 1 is one new result per cycle after
    filling; latency includes line/window fill plus arithmetic pipeline
    stages. Boundary slots and initial/final fill/drain reduce utilization;
    the interior fraction is (rows-2)*(columns-2)/(rows*columns) for rows,
    columns >= 3. Preserve boundaries through a suitably delayed bypass.
    """
    rows, columns = _shape(grid)
    result = [row[:] for row in grid]
    for i in range(1, rows - 1):
        for j in range(1, columns - 1):
            result[i][j] = (grid[i][j] + grid[i - 1][j] + grid[i + 1][j]
                            + grid[i][j - 1] + grid[i][j + 1]) / 5.0
    return result


def _run_checks() -> None:
    """Check hand-computed answers, edge cases, and invalid inputs."""
    assert vector_add([1, 2, 3], [4, 5, 6]) == [5, 7, 9]
    assert vector_add([], []) == []
    A = [[1, 2, 3], [4, 5, 6]]
    B = [[7, 8], [9, 10], [11, 12]]
    expected = [[58, 64], [139, 154]]
    assert matmul(A, B) == expected
    assert matmul([], []) == []
    for size in (1, 2, 3, 8):
        assert matmul_tiled(A, B, size) == expected
        assert matmul_tiled([], [], size) == []
        for matrix in ([], [[7]], [[1, 2, 3]], [[1], [2], [3]], A):
            assert transpose_blocked(matrix, size) == transpose(matrix)
            assert transpose_blocked(transpose_blocked(matrix, size), size) == matrix
            assert transpose(transpose(matrix)) == matrix
    assert transpose(A) == [[1, 4], [2, 5], [3, 6]]
    for values, total in (([], 0), ([7], 7), ([1, 2, 3, 4, 5], 15),
                          ([1, 2, 3, 4], 10), ([-3, 2, 1], 0)):
        assert sequential_sum(values) == total
        assert tree_sum(values) == total
    grid = [[1, 2, 3, 4], [5, 20, 7, 8], [9, 10, 30, 12], [13, 14, 15, 16]]
    original = [row[:] for row in grid]
    assert five_point_stencil(grid) == [
        [1, 2, 3, 4], [5, 8.8, 13.6, 8], [9, 16.6, 14.8, 12], [13, 14, 15, 16]
    ]
    assert grid == original
    for small in ([], [[1]], [[1, 2, 3]], [[1], [2], [3]], [[1, 2], [3, 4]]):
        copied = five_point_stencil(small)
        assert copied == small and copied is not small
        assert all(a is not b for a, b in zip(copied, small))
    invalid_calls = [
        (vector_add, ([1], [])),
        (matmul, ([[1, 2]], [[3, 4]])),
        (matmul_tiled, ([[1, 2]], [[3, 4]])),
        (transpose, ([[1], [2, 3]],)),
        (transpose, ([[], []],)),
        (five_point_stencil, ([[1], []],)),
    ]
    for size in (0, -1, 1.5, True):
        invalid_calls.extend([
            (matmul_tiled, (A, B, size)),
            (transpose_blocked, (A, size)),
        ])
    for function, arguments in invalid_calls:
        try:
            function(*arguments)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{function.__name__} accepted invalid input")
    print("All five exercises passed the example and edge-case checks.")


if __name__ == "__main__":
    _run_checks()
