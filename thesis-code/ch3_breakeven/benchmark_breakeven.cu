/*
 * benchmark_breakeven.cu
 *
 * Measures CUDA kernel launch overhead for chains of N trivial kernels,
 * with and without CUDA graph capture. Outputs CSV to stdout.
 *
 * Compile: nvcc -O2 -o benchmark_breakeven benchmark_breakeven.cu
 * Run:     ./benchmark_breakeven | tee breakeven_data.csv
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <time.h>

#define VEC_SIZE 128
#define WARMUP   50
#define REPEATS  1000

#define CHECK(err) do {                                                      \
    cudaError_t _e = (err);                                                  \
    if (_e != cudaSuccess) {                                                 \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                           \
                __FILE__, __LINE__, cudaGetErrorString(_e));                 \
        exit(1);                                                             \
    }                                                                        \
} while (0)

__global__ void daxpy(int n, double alpha, const double* x, double* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += alpha * x[i];
}

static double cpu_us() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec * 1e-3;
}

int main() {
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    fprintf(stderr, "Device : %s\n", prop.name);
    fprintf(stderr, "VecSize: %d   Warmup: %d   Repeats: %d\n",
            VEC_SIZE, WARMUP, REPEATS);

    /* header */
    printf("# N,t_chain_us,t_per_kernel_us,t_inst_us,t_graph_launch_us,r_break\n");

    double *d_x, *d_y;
    CHECK(cudaMalloc(&d_x, VEC_SIZE * sizeof(double)));
    CHECK(cudaMalloc(&d_y, VEC_SIZE * sizeof(double)));
    double h[VEC_SIZE];
    for (int i = 0; i < VEC_SIZE; i++) h[i] = 1.0;
    CHECK(cudaMemcpy(d_x, h, VEC_SIZE * sizeof(double), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_y, h, VEC_SIZE * sizeof(double), cudaMemcpyHostToDevice));

    const double alpha  = 0.5;
    const int    blocks = (VEC_SIZE + 127) / 128;

    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    cudaEvent_t ev_start, ev_end;
    CHECK(cudaEventCreate(&ev_start));
    CHECK(cudaEventCreate(&ev_end));

    /* force JIT compilation before any timing */
    daxpy<<<blocks, 128, 0, stream>>>(VEC_SIZE, alpha, d_x, d_y);
    CHECK(cudaStreamSynchronize(stream));

    const int N_VALUES[] = {1, 2, 5, 10, 17, 22};
    const int N_COUNT    = (int)(sizeof(N_VALUES) / sizeof(N_VALUES[0]));

    for (int vi = 0; vi < N_COUNT; vi++) {
        int N = N_VALUES[vi];

        /* ---- no-graph chain ---- */
        for (int r = 0; r < WARMUP; r++)
            for (int k = 0; k < N; k++)
                daxpy<<<blocks, 128, 0, stream>>>(VEC_SIZE, alpha, d_x, d_y);
        CHECK(cudaStreamSynchronize(stream));

        CHECK(cudaEventRecord(ev_start, stream));
        for (int r = 0; r < REPEATS; r++)
            for (int k = 0; k < N; k++)
                daxpy<<<blocks, 128, 0, stream>>>(VEC_SIZE, alpha, d_x, d_y);
        CHECK(cudaEventRecord(ev_end, stream));
        CHECK(cudaEventSynchronize(ev_end));

        float ms_chain;
        CHECK(cudaEventElapsedTime(&ms_chain, ev_start, ev_end));
        double t_chain = (double)ms_chain * 1e3 / REPEATS;  /* μs per N-chain */

        /* ---- graph capture ---- */
        cudaGraph_t     graph;
        cudaGraphExec_t graphExec;

        CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        for (int k = 0; k < N; k++)
            daxpy<<<blocks, 128, 0, stream>>>(VEC_SIZE, alpha, d_x, d_y);
        CHECK(cudaStreamEndCapture(stream, &graph));

        /* instantiation — CPU-timed (CUDA 12: 3-arg form) */
        double t0 = cpu_us();
        CHECK(cudaGraphInstantiate(&graphExec, graph, 0));
        double t_inst = cpu_us() - t0;

        /* warmup graph launches */
        for (int r = 0; r < WARMUP; r++)
            CHECK(cudaGraphLaunch(graphExec, stream));
        CHECK(cudaStreamSynchronize(stream));

        /* timed graph launches */
        CHECK(cudaEventRecord(ev_start, stream));
        for (int r = 0; r < REPEATS; r++)
            CHECK(cudaGraphLaunch(graphExec, stream));
        CHECK(cudaEventRecord(ev_end, stream));
        CHECK(cudaEventSynchronize(ev_end));

        float ms_graph;
        CHECK(cudaEventElapsedTime(&ms_graph, ev_start, ev_end));
        double t_graph = (double)ms_graph * 1e3 / REPEATS;  /* μs per graph launch */

        double denom  = t_chain - t_graph;
        double r_break = (denom > 0.001) ? t_inst / denom : 1e9;

        printf("%d,%.3f,%.3f,%.3f,%.3f,%.3f\n",
               N, t_chain, t_chain / N, t_inst, t_graph, r_break);
        fflush(stdout);

        fprintf(stderr, "N=%2d: chain=%.2f us  inst=%.2f us  launch=%.2f us  R_break=%.2f\n",
                N, t_chain, t_inst, t_graph, r_break);

        CHECK(cudaGraphExecDestroy(graphExec));
        CHECK(cudaGraphDestroy(graph));
    }

    CHECK(cudaFree(d_x));
    CHECK(cudaFree(d_y));
    CHECK(cudaStreamDestroy(stream));
    CHECK(cudaEventDestroy(ev_start));
    CHECK(cudaEventDestroy(ev_end));
    return 0;
}
