/*
 * nbody_cuda.cu - Mo phong N-body tren GPU bang CUDA.
 *
 * Hai kernel tinh gia toc:
 *   naive : moi thread = 1 vat the, doc truc tiep N vi tri tu global memory.
 *   tiled : thuat toan GPU Gems 3, chuong 31 ("Fast N-Body Simulation with CUDA"):
 *           chia N vat the thanh cac tile kich thuoc p = blockDim.x; moi block lan luot
 *           nap tung tile vao shared memory (moi thread nap 1 vat the), dong bo, roi
 *           moi thread tinh p tuong tac voi du lieu trong shared memory.
 *           => moi vi tri chi doc tu global memory N/p lan thay vi N lan.
 *
 * Bien dich:  nvcc -O3 -arch=native -use_fast_math -Xcompiler -fopenmp src/nbody_cuda.cu -o bin/nbody_cuda
 * Vi du:      ./bin/nbody_cuda -n 65536 -s 10 -k tiled -b 256
 *             ./bin/nbody_cuda -n 4096 -s 5 -verify
 */
#include <cuda_runtime.h>
#include "nbody_common.h"
#include "nbody_cpu_kernels.h"

#define CUDA_CHECK(call)                                                              \
    do {                                                                              \
        cudaError_t err__ = (call);                                                   \
        if (err__ != cudaSuccess) {                                                   \
            fprintf(stderr, "Loi CUDA %s:%d: %s\n", __FILE__, __LINE__,               \
                    cudaGetErrorString(err__));                                       \
            exit(1);                                                                  \
        }                                                                             \
    } while (0)

/* Tuong tac giua vat the i va vat the j (GPU Gems 3, Listing 31-1). ~20 flop. */
__device__ __forceinline__ float3 bodyBodyInteraction(float4 bi, float4 bj, float3 ai, float eps2)
{
    float3 r;
    r.x = bj.x - bi.x;                                   /* 3 flop */
    r.y = bj.y - bi.y;
    r.z = bj.z - bi.z;
    float distSqr = r.x * r.x + r.y * r.y + r.z * r.z + eps2;   /* 6 flop */
    float invDist = rsqrtf(distSqr);                     /* 2 flop */
    float invDist3 = invDist * invDist * invDist;        /* 2 flop */
    float s = bj.w * invDist3;                           /* 1 flop */
    ai.x += r.x * s;                                     /* 6 flop */
    ai.y += r.y * s;
    ai.z += r.z * s;
    return ai;
}

/* ---------------- Kernel 1: naive (chi dung global memory) ---------------- */
__global__ void accNaive(const float4 *__restrict__ pos, float4 *__restrict__ acc, int n, float eps2)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float4 bi = pos[i];
    float3 a = make_float3(0.0f, 0.0f, 0.0f);
    for (int j = 0; j < n; j++) a = bodyBodyInteraction(bi, pos[j], a, eps2);
    acc[i] = make_float4(a.x, a.y, a.z, 0.0f);
}

/* ---------------- Kernel 2: tiled + shared memory (GPU Gems 3, Listing 31-2/31-3) ---------------- */
__global__ void accTiled(const float4 *__restrict__ pos, float4 *__restrict__ acc, int n, float eps2)
{
    extern __shared__ float4 shPos[];
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    /* Khong 'return' som: moi thread phai tham gia __syncthreads() va nap tile. */
    const float4 bi = (i < n) ? pos[i] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float3 a = make_float3(0.0f, 0.0f, 0.0f);

    for (int tile = 0; tile * blockDim.x < n; tile++) {
        int j = tile * blockDim.x + threadIdx.x;
        /* Phan tu dem (j >= n) co khoi luong 0 => khong dong gop luc. */
        shPos[threadIdx.x] = (j < n) ? pos[j] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        __syncthreads();
#pragma unroll 8
        for (int k = 0; k < blockDim.x; k++) a = bodyBodyInteraction(bi, shPos[k], a, eps2);
        __syncthreads();
    }
    if (i < n) acc[i] = make_float4(a.x, a.y, a.z, 0.0f);
}

/* ---------------- Cap nhat van toc, vi tri (symplectic Euler) ---------------- */
__global__ void integrateBodies(float4 *__restrict__ pos, float4 *__restrict__ vel,
                                const float4 *__restrict__ acc, int n, float dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float4 p = pos[i], v = vel[i], a = acc[i];
    v.x += a.x * dt;  v.y += a.y * dt;  v.z += a.z * dt;
    p.x += v.x * dt;  p.y += v.y * dt;  p.z += v.z * dt;
    pos[i] = p;
    vel[i] = v;
}

static void launchAcc(int tiled, const float4 *dPos, float4 *dAcc, int n, float eps2, int block)
{
    int grid = (n + block - 1) / block;
    if (tiled) accTiled<<<grid, block, block * sizeof(float4)>>>(dPos, dAcc, n, eps2);
    else       accNaive<<<grid, block>>>(dPos, dAcc, n, eps2);
}

int main(int argc, char **argv)
{
    nb_params p;
    nb_parse(argc, argv, &p, 1);
    int tiled;
    if (!strcmp(p.mode, "tiled")) tiled = 1;
    else if (!strcmp(p.mode, "naive")) tiled = 0;
    else { fprintf(stderr, "Kernel khong hop le: %s (naive | tiled)\n", p.mode); return 1; }

    int dev = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    if (p.block > prop.maxThreadsPerBlock) {
        fprintf(stderr, "Block %d vuot qua gioi han %d\n", p.block, prop.maxThreadsPerBlock);
        return 1;
    }

    printf("== N-body CUDA | GPU: %s (SM %d.%d, %d SMs) | kernel: %s | N = %d | steps = %d | block = %d ==\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount, p.mode, p.n, p.steps, p.block);

    const size_t bytes = sizeof(vec4f) * (size_t)p.n;
    const float eps2 = p.eps * p.eps;
    vec4f *hPos = (vec4f *)malloc(bytes), *hVel = (vec4f *)malloc(bytes);
    vec4f *hPos0 = (vec4f *)malloc(bytes), *hVel0 = (vec4f *)malloc(bytes);
    if (!hPos || !hVel || !hPos0 || !hVel0) { fprintf(stderr, "Khong du bo nho\n"); return 1; }
    nb_init_plummer(hPos, hVel, p.n, p.seed);
    memcpy(hPos0, hPos, bytes);
    memcpy(hVel0, hVel, bytes);

    double ke0 = 0, pe0 = 0;
    if (!p.no_energy) nb_energy(hPos, hVel, p.n, p.eps, &ke0, &pe0);

    float4 *dPos, *dVel, *dAcc;
    CUDA_CHECK(cudaMalloc(&dPos, bytes));
    CUDA_CHECK(cudaMalloc(&dVel, bytes));
    CUDA_CHECK(cudaMalloc(&dAcc, bytes));

    cudaEvent_t evH2D0, evH2D1, ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&evH2D0));
    CUDA_CHECK(cudaEventCreate(&evH2D1));
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));

    CUDA_CHECK(cudaEventRecord(evH2D0));
    CUDA_CHECK(cudaMemcpy(dPos, hPos, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dVel, hVel, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(evH2D1));

    /* Warm-up: chi tinh gia toc, khong cap nhat trang thai => khong anh huong ket qua. */
    launchAcc(tiled, dPos, dAcc, p.n, eps2, p.block);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    FILE *out = NULL;
    if (p.out) {
        out = fopen(p.out, "w");
        if (!out) { fprintf(stderr, "Khong mo duoc %s\n", p.out); return 1; }
        nb_write_header(out);
        nb_write_frame(out, 0, hPos, p.n);
    }

    const int grid = (p.n + p.block - 1) / p.block;
    float msTotal = 0.0f, ms;
    CUDA_CHECK(cudaEventRecord(ev0));
    for (int s = 1; s <= p.steps; s++) {
        launchAcc(tiled, dPos, dAcc, p.n, eps2, p.block);
        integrateBodies<<<grid, p.block>>>(dPos, dVel, dAcc, p.n, p.dt);
        if (out && s % p.every == 0) {
            /* Khong tinh thoi gian sao chep + ghi file vao thoi gian tinh toan. */
            CUDA_CHECK(cudaEventRecord(ev1));
            CUDA_CHECK(cudaEventSynchronize(ev1));
            CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
            msTotal += ms;
            CUDA_CHECK(cudaMemcpy(hPos, dPos, bytes, cudaMemcpyDeviceToHost));
            nb_write_frame(out, s, hPos, p.n);
            CUDA_CHECK(cudaEventRecord(ev0));
        }
    }
    CUDA_CHECK(cudaEventRecord(ev1));
    CUDA_CHECK(cudaEventSynchronize(ev1));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
    msTotal += ms;
    if (out) fclose(out);

    double t0 = nb_wtime();
    CUDA_CHECK(cudaMemcpy(hPos, dPos, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hVel, dVel, bytes, cudaMemcpyDeviceToHost));
    double d2hSec = nb_wtime() - t0;
    float h2dMs;
    CUDA_CHECK(cudaEventElapsedTime(&h2dMs, evH2D0, evH2D1));

    nb_report("cuda", &p, p.block, msTotal / 1000.0);
    printf("  Sao chep H2D / D2H  : %.3f ms / %.3f ms\n", h2dMs, 1e3 * d2hSec);

    if (!p.no_energy) {
        double ke1, pe1;
        nb_energy(hPos, hVel, p.n, p.eps, &ke1, &pe1);
        double e0 = ke0 + pe0, e1 = ke1 + pe1;
        printf("  Nang luong E0 = %.6f, E1 = %.6f, |dE/E0| = %.3e\n", e0, e1, fabs((e1 - e0) / e0));
    }

    /* -------- So sanh voi ban CPU (OpenMP) cung trang thai ban dau -------- */
    if (p.verify) {
        vec4f *acc = (vec4f *)malloc(bytes);
        double tc = nb_run_cpu(hPos0, hVel0, acc, p.n, p.steps, p.dt, p.eps, 1, NULL, 1);
        double err = nb_max_rel_diff(hPos, hPos0, p.n);
        printf("[verify] CPU OpenMP %.4f s | GPU %.4f s | speedup %.1fx | sai so tuong doi max = %.3e -> %s\n",
               tc, msTotal / 1000.0, tc / (msTotal / 1000.0), err, err < 1e-3 ? "PASS" : "FAIL");
        free(acc);
    }

    CUDA_CHECK(cudaFree(dPos));
    CUDA_CHECK(cudaFree(dVel));
    CUDA_CHECK(cudaFree(dAcc));
    cudaEventDestroy(evH2D0); cudaEventDestroy(evH2D1);
    cudaEventDestroy(ev0); cudaEventDestroy(ev1);
    free(hPos); free(hVel); free(hPos0); free(hVel0);
    return 0;
}
