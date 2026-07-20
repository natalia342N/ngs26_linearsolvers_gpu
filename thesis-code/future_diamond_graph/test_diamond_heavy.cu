// test_diamond_heavy.cu
//
// Standalone diamond graph benchmark — no NGSolve dependency.
// Compares sequential vs diamond execution of two independent heavy branches.
//
// Build:
//   nvcc -O2 -arch=sm_90 test_diamond_heavy.cu -o test_diamond_heavy
//
// Tune NBLOCKS to be well under half the SM count so both branches
// fit simultaneously. ITERS controls branch weight — aim for ~2-5 ms each.

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

// ── tuneable constants ────────────────────────────────────────────────────────
#ifndef N
#define N       (1 << 20)   // 1M elements per branch (8 MB per array)
#endif
#ifndef ITERS
#define ITERS   200         // sqrt iterations per element — increase for heavier branches
#endif
#ifndef NBLOCKS
#define NBLOCKS 64          // keep well under SM_count/2 so both branches can run in parallel
#endif
#define NTHREADS  256
#define TIMING_RUNS 20
// ─────────────────────────────────────────────────────────────────────────────


// Compute-bound kernel: each thread runs ITERS sqrt operations.
// Using a recurrence so the compiler cannot hoist the loop out.
__global__ void heavy_work(double* __restrict__ y,
                            const double* __restrict__ x,
                            int n, int iters)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x)
    {
        double v = x[i];
        #pragma unroll 1
        for (int it = 0; it < iters; it++)
            v = sqrt(v + 1.0001);   // always positive, no NaN
        y[i] = v;
    }
}

__global__ void set_val(double* d, double val)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) d[0] = val;
}


// ── graph capture helpers ─────────────────────────────────────────────────────

static cudaGraph_t capture_heavy(cudaStream_t s,
                                  double* y, const double* x,
                                  int n, int iters)
{
    cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal);
    heavy_work<<<NBLOCKS, NTHREADS, 0, s>>>(y, x, n, iters);
    cudaGraph_t g;
    cudaStreamEndCapture(s, &g);
    return g;
}

static cudaGraph_t capture_trivial(cudaStream_t s, double* d, double val)
{
    cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal);
    set_val<<<1, 1, 0, s>>>(d, val);
    cudaGraph_t g;
    cudaStreamEndCapture(s, &g);
    return g;
}


// ── graph builders ────────────────────────────────────────────────────────────

//   g_pre → { g_x ∥ g_r } → g_post
static cudaGraphExec_t build_diamond(cudaGraph_t g_pre, cudaGraph_t g_x,
                                      cudaGraph_t g_r,  cudaGraph_t g_post)
{
    cudaGraph_t g;
    cudaGraphCreate(&g, 0);

    cudaGraphNode_t n_pre, n_x, n_r, n_post;
    cudaGraphAddChildGraphNode(&n_pre,  g, nullptr,  0, g_pre);
    cudaGraphAddChildGraphNode(&n_x,    g, &n_pre,   1, g_x);
    cudaGraphAddChildGraphNode(&n_r,    g, &n_pre,   1, g_r);   // no edge to n_x
    cudaGraphNode_t join[2] = {n_x, n_r};
    cudaGraphAddChildGraphNode(&n_post, g, join,      2, g_post);

    cudaGraphExec_t exec;
    cudaGraphInstantiate(&exec, g, NULL, NULL, 0);
    cudaGraphDestroy(g);
    return exec;
}

//   g_pre → g_x → g_r → g_post  (fully sequential, same work)
static cudaGraphExec_t build_sequential(cudaGraph_t g_pre, cudaGraph_t g_x,
                                         cudaGraph_t g_r,  cudaGraph_t g_post)
{
    cudaGraph_t g;
    cudaGraphCreate(&g, 0);

    cudaGraphNode_t n_pre, n_x, n_r, n_post;
    cudaGraphAddChildGraphNode(&n_pre,  g, nullptr, 0, g_pre);
    cudaGraphAddChildGraphNode(&n_x,    g, &n_pre,  1, g_x);
    cudaGraphAddChildGraphNode(&n_r,    g, &n_x,    1, g_r);    // n_r waits for n_x
    cudaGraphAddChildGraphNode(&n_post, g, &n_r,    1, g_post);

    cudaGraphExec_t exec;
    cudaGraphInstantiate(&exec, g, NULL, NULL, 0);
    cudaGraphDestroy(g);
    return exec;
}


// ── timing ────────────────────────────────────────────────────────────────────

static float time_graph(cudaGraphExec_t exec, cudaStream_t s, int runs)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, s);
    for (int i = 0; i < runs; i++)
        cudaGraphLaunch(exec, s);
    cudaEventRecord(stop, s);
    cudaStreamSynchronize(s);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms / runs;
}


// ── main ──────────────────────────────────────────────────────────────────────

int main()
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device : %s  (%d SMs)\n", prop.name, prop.multiProcessorCount);
    printf("N      : %d elements per branch (%.1f MB)\n",
           N, N * sizeof(double) / 1e6);
    printf("ITERS  : %d sqrt ops per element\n", ITERS);
    printf("NBLOCKS: %d  (%.0f%% of SMs per branch)\n\n",
           NBLOCKS, 100.0 * NBLOCKS / prop.multiProcessorCount);

    // allocate independent arrays for each branch
    double *x_in, *x_out, *r_in, *r_out, *d_scalar;
    cudaMalloc(&x_in,    N * sizeof(double));
    cudaMalloc(&x_out,   N * sizeof(double));
    cudaMalloc(&r_in,    N * sizeof(double));
    cudaMalloc(&r_out,   N * sizeof(double));
    cudaMalloc(&d_scalar, sizeof(double));
    cudaMemset(x_in, 0, N * sizeof(double));
    cudaMemset(r_in, 0, N * sizeof(double));

    // capture the four sub-graphs on a single stream (just as in NGSolve)
    cudaStream_t cap_stream;
    cudaStreamCreate(&cap_stream);

    cudaGraph_t g_pre  = capture_trivial(cap_stream, d_scalar, 1.0);
    cudaGraph_t g_x    = capture_heavy  (cap_stream, x_out, x_in, N, ITERS);
    cudaGraph_t g_r    = capture_heavy  (cap_stream, r_out, r_in, N, ITERS);
    cudaGraph_t g_post = capture_trivial(cap_stream, d_scalar, 0.0);

    cudaGraphExec_t seq_exec = build_sequential(g_pre, g_x, g_r, g_post);
    cudaGraphExec_t dia_exec = build_diamond   (g_pre, g_x, g_r, g_post);

    cudaStream_t timing_stream;
    cudaStreamCreate(&timing_stream);

    // warm up both
    cudaGraphLaunch(seq_exec, timing_stream); cudaStreamSynchronize(timing_stream);
    cudaGraphLaunch(dia_exec, timing_stream); cudaStreamSynchronize(timing_stream);

    float seq_ms = time_graph(seq_exec, timing_stream, TIMING_RUNS);
    float dia_ms = time_graph(dia_exec, timing_stream, TIMING_RUNS);

    printf("Sequential : %.3f ms\n", seq_ms);
    printf("Diamond    : %.3f ms\n", dia_ms);
    printf("Speedup    : %.2fx\n",   seq_ms / dia_ms);
    printf("\n(ideal speedup if branches fully overlap: ~%.1fx)\n",
           seq_ms / (seq_ms - dia_ms + dia_ms));  // just echoes 1x, see note below

    // For reference: if branches are ~equal weight and fully overlap,
    // expected diamond time ≈ seq_time / 2.

    // cleanup
    cudaGraphExecDestroy(seq_exec);
    cudaGraphExecDestroy(dia_exec);
    cudaGraphDestroy(g_pre);
    cudaGraphDestroy(g_x);
    cudaGraphDestroy(g_r);
    cudaGraphDestroy(g_post);
    cudaStreamDestroy(cap_stream);
    cudaStreamDestroy(timing_stream);
    cudaFree(x_in); cudaFree(x_out);
    cudaFree(r_in); cudaFree(r_out);
    cudaFree(d_scalar);

    return 0;
}
