// gpu_trunc_norm.cu
// Author: Nick Ulle

#include <stdio.h>
#include <math.h>
#include <curand_kernel.h>

#define MAX_ITER 100

__device__ float tail_normal(
    float mean, 
    float sd, 
    float a, 
    curandState_t *rng_state)
{
    a = (a - mean) / sd;
    float tail = 1;
    // The algorithm samples right tail N(0, 1), so negate everything if left
    // tail samples are needed.
    if (a < 0) tail = -1;
    a *= tail;

    float z = 0;
    int i;
    for (i = 0; i < MAX_ITER; i++)
    {
        // Generate z ~ EXP(alpha) + a.
        float alpha = (a + sqrtf(powf(a, 2) + 4)) / 2;
        float u = curand_uniform(rng_state);
        z = -logf(1 - u) / alpha + a;

        // Compute g(z).
        float gz = expf(-powf(z - alpha, 2) / 2);

        // Generate u and test acceptance.
        u = curand_uniform(rng_state);

        if (u <= gz) break;
    }
    return sd * tail * z + mean;
}

extern "C" {
__global__ void gpu_trunc_normal(
    int n, 
    float *mean, 
    float *sd,
    float *a,
    float *b,
    float *result)
{
    // Get block and thread numbers.
    int block = blockIdx.x + blockIdx.y * gridDim.x;
    int thread = threadIdx.x + threadIdx.y * blockDim.x
        + threadIdx.z * (blockDim.y * blockDim.x);
    int block_size = blockDim.x * blockDim.y * blockDim.z;
    int idx = thread + block * block_size;

    // Only compute if the index is less than the result length.
    if (idx < n)
    {
        // Initialize the RNG. This could be done in a separate kernel.
        curandState_t rng_state;
        curand_init(109 + 126*block, thread, 0, &rng_state);

        // Get truncated normal value. Use C. Robert's algorithm if possible.
        float draw = 0;
        if (!isfinite(a[idx]) && b[idx] <= mean[idx]) 
        {
            draw = tail_normal(mean[idx], sd[idx], b[idx], &rng_state);
        } else if (!isfinite(b[idx]) && a[idx] >= mean[idx]) 
        {
            /*
            printf("RT %i has a = %f, b = %f, mean = %f\n", idx, a[idx], \
                   b[idx], mean[idx]);
            */
            draw = tail_normal(mean[idx], sd[idx], a[idx], &rng_state);
        } else 
        {
            /*
            printf("N %i has a = %f, b = %f, mean = %f\n", idx, a[idx], \
                   b[idx], mean[idx]);
            */
            for (int i = 0; i < MAX_ITER; i++)
            {
                draw = sd[idx] * curand_normal(&rng_state) + mean[idx];

                if (a[idx] <= draw && draw <= b[idx]) break;
            }
        } // end if
        
        result[idx] = draw;
    }
}
} // end extern

