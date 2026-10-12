"""CPU vs MPS matmul benchmark. Edit SIZES / REPEATS / DTYPE and re-run.

    uv run python bench_matmul.py
"""

import time

import torch

SIZES = [1000, 2000, 4000, 8000, 12000]
REPEATS = 5
DTYPE = torch.float32


def bench(x, sync):
    x @ x  # warm-up
    sync()
    times = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        x @ x
        sync()
        times.append(time.perf_counter() - t0)
    return min(times), sum(times) / len(times)


def main():
    has_mps = torch.backends.mps.is_available()
    print(f"torch {torch.__version__} | dtype {DTYPE} | repeats {REPEATS} | mps {has_mps}")
    print(f"{'N':>6} {'cpu best':>10} {'cpu avg':>10} {'mps best':>10} {'mps avg':>10} {'speedup':>8} {'mps TFLOPS':>11}")
    for n in SIZES:
        x = torch.randn(n, n, dtype=DTYPE)
        cpu_best, cpu_avg = bench(x, lambda: None)
        if has_mps:
            xm = x.to("mps")
            mps_best, mps_avg = bench(xm, torch.mps.synchronize)
            tflops = 2 * n**3 / mps_best / 1e12
            print(f"{n:>6} {cpu_best:>10.4f} {cpu_avg:>10.4f} {mps_best:>10.4f} {mps_avg:>10.4f} {cpu_best / mps_best:>7.1f}x {tflops:>11.2f}")
            del xm
        else:
            print(f"{n:>6} {cpu_best:>10.4f} {cpu_avg:>10.4f}")


if __name__ == "__main__":
    main()
