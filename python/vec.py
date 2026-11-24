

def vector_add(a: list[float], b: list[float]) -> list[float]:
    length = max(len(a), len(b))
    c = []
    for i in range(length): 
        c.append(a[i] + b[i])
    return c
    
# per element, 8 bytes and 1 op per element 
# at 100GB/s, 100/8 = 12.5Gelement/s 
# 12.5Gop/s is the upper bound perf, more compute performance will be idle 
# SIMD will be useful for this . If a and b are well-coelesced, it is good 

def main(): 
    a = [1.0, 2, 3]
    b = [4.0, 5, 6]
    print(vector_add(a, b))
    
main()


# REVIEW: 6/10 against README.md exercise 1.
#
# Code: 3/4
# - Correct elementwise addition for equal lengths; [] + [] returns [].
# - Unequal lengths cause an accidental IndexError. Explicitly check lengths
#   and raise ValueError before the loop, then use length = len(a).
# - Time is O(n); the output requires O(n) space.
#
# Memory traffic and throughput: 1/3
# - Correct: one floating-point addition per element.
# - Count the output store too: 4 bytes from a + 4 from b + 4 to c = 12 bytes.
# - At 100 GB/s: 100e9 / 12 = approximately 8.33e9 elements/s = 8.33 GFLOP/s.
# - Arithmetic intensity is 1/12 FLOP/byte. These numbers assume contiguous
#   FP32 arrays and no write allocation, not actual Python list storage.
#
# Bottleneck explanation: 1/1
# - Correct: once memory bandwidth is saturated, extra compute cannot improve
#   throughput. The roofline bound is min(peak FLOP/s, bandwidth / 12).
#
# SIMD and alignment: 1/2
# - Correct direction, but explain that SIMD adds multiple elements per
#   instruction, reducing instruction overhead and helping saturate bandwidth.
# - Contiguous, cache-line-aligned arrays help avoid split-line accesses;
#   handle any elements left after the full SIMD vectors with a tail loop/mask.
# - Coalescing usually describes neighboring GPU threads accessing neighboring
#   addresses. For CPU SIMD, discuss contiguous vector loads/stores and alignment.
# - This Python loop does not itself explicitly use SIMD instructions.
#
# Minor cleanup: main() computes the same sum twice and discards the first.
# Use an if __name__ == "__main__": guard to avoid running main on import.
# Verified: equal-length and empty cases pass; unequal-length cases raise
# IndexError. Original answer above is preserved for comparison.
