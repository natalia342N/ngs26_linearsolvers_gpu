import ngsolve.ngscuda as ngscuda

print("=== COMPUTE-BOUND (sqrt loop) — CudaGraph/CudaDiamondGraph/ngs_cuda_stream ===")
print("N=1M elements, NBLOCKS=64 (48% of H100 SMs per branch)\n")

for iters in [200, 1000, 5000]:
    seq_ms, dia_ms = ngscuda.DiamondHeavyBenchmark(
        n=1 << 20, nblocks=64, iters=iters, nruns=20)
    print(f"ITERS={iters:5d}  sequential={seq_ms:.3f} ms  "
          f"diamond={dia_ms:.3f} ms  speedup={seq_ms/dia_ms:.2f}x")

print()
print("=== MEMORY-BOUND (DAXPY) — more representative of NGSolve vector ops ===")
print("N=1M elements, NBLOCKS=64, varying repeats\n")

for repeats in [50, 200, 1000]:
    seq_ms, dia_ms = ngscuda.DiamondDaxpyBenchmark(
        n=1 << 20, nblocks=64, repeats=repeats, nruns=20)
    print(f"REPEATS={repeats:5d}  sequential={seq_ms:.3f} ms  "
          f"diamond={dia_ms:.3f} ms  speedup={seq_ms/dia_ms:.2f}x")

print()
print("=== cuBLAS Ddot — two independent dot products (most relevant to CG inner products) ===")
print("POINTER_MODE_DEVICE, varying N\n")

for n in [1 << 14, 1 << 17, 1 << 20, 1 << 23]:
    seq_ms, dia_ms = ngscuda.DiamondDotBenchmark(n=n, nruns=50)
    print(f"N={n:>9d}  sequential={seq_ms*1000:.2f} us  "
          f"diamond={dia_ms*1000:.2f} us  speedup={seq_ms/dia_ms:.2f}x")
