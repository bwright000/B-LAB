"""Measured GPU memory bandwidth (device-to-device copy) and FP16 matmul throughput. Median of 10."""
import statistics, cupy as cp

def time_ms(fn, n=10):
    s, e = cp.cuda.Event(), cp.cuda.Event(); fn(); cp.cuda.Device().synchronize()
    out = []
    for _ in range(n):
        s.record(); fn(); e.record(); e.synchronize(); out.append(cp.cuda.get_elapsed_time(s, e))
    return statistics.median(out)

n = 2 * 1024**3 // 2                      # 2 GiB of fp16
a = cp.ones(n, dtype=cp.float16); b = cp.empty_like(a)
ms = time_ms(lambda: cp.copyto(b, a))
print(f"copy bandwidth : {2 * a.nbytes / (ms / 1e3) / 1e9:6.0f} GB/s  (read+write, 2 GiB buffer)")
del a, b
for sz in (4096, 8192):
    x = cp.random.rand(sz, sz).astype(cp.float16); y = cp.random.rand(sz, sz).astype(cp.float16)
    ms = time_ms(lambda: x @ y)
    print(f"fp16 GEMM {sz}^3: {2 * sz**3 / (ms / 1e3) / 1e12:6.1f} TFLOPS")
