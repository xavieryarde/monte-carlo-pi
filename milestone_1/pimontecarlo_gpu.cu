#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "pimontecarlo.h"

namespace
{
    __global__
    void setupKernel(curandState *states)
    {
        long long tid = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
        curand_init(12345, tid, 0, &states[tid]);
    }

    __global__ 
    void estimatePIKernel(long long TP, unsigned long long *PIC, curandState *states)
    {
        long long tid = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
        long long stride = static_cast<long long>(gridDim.x) * blockDim.x;

        curandState gen = states[tid];

        unsigned long long hits = 0;

        for (long long i = tid; i < TP; i += stride)
        {
            
            float x = curand_uniform(&gen) * 2.0f - 1.0f;
            float y = curand_uniform(&gen) * 2.0f - 1.0f;

            if ((x * x) + (y * y) <= 1.0f)
            {
                hits++;
            }
        }

        // states[tid] = gen;

        __shared__ unsigned long long sData[256];
        sData[threadIdx.x] = hits;
        __syncthreads();

        for (int s = blockDim.x / 2; s > 0; s >>= 1)
        {
            if (threadIdx.x < s)
                sData[threadIdx.x] += sData[threadIdx.x + s];
            __syncthreads();
        }

        if (threadIdx.x == 0)
            atomicAdd(PIC, sData[0]);
    }

    double estimatePIMonteCarloImpl(long long TP, double &elapsedSeconds)
    {
        const int threadsPerBlock = 256;
        const int numBlocks = static_cast<int>(std::min<long long>((TP + threadsPerBlock - 1) / threadsPerBlock, 1024));

        static curandState *d_states = nullptr;
        if (d_states == nullptr)
        {
            cudaMalloc(&d_states, 1024 * threadsPerBlock * sizeof(curandState));
            setupKernel<<<1024, threadsPerBlock>>>(d_states);
            cudaDeviceSynchronize();
        }
        
        unsigned long long *d_PIC;
        unsigned long long PIC = 0;

        cudaMalloc(&d_PIC, sizeof(unsigned long long));
        cudaMemset(d_PIC, 0, sizeof(unsigned long long));

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        estimatePIKernel<<<numBlocks, threadsPerBlock>>>(TP, d_PIC, d_states);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float milliseconds = 0.0f;
        cudaEventElapsedTime(&milliseconds, start, stop);
        elapsedSeconds = milliseconds / 1000.0;

        cudaMemcpy(&PIC, d_PIC, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(d_PIC);

        double pi_estimate = 4.0 * static_cast<double>(PIC) / static_cast<double>(TP);

        return pi_estimate;
    }
}

double estimatePIMonteCarloGPU(long long TP, double &elapsedSeconds)
{
    // Warm up the GPU to avoid measuring initialization overhead in the first run

    static bool warmedUp = false;
    if (!warmedUp)
    {
        double ignored = 0.0;
        estimatePIMonteCarloImpl(1, ignored);
        warmedUp = true;
    }

    return estimatePIMonteCarloImpl(TP, elapsedSeconds);
}
