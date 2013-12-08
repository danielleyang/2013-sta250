// gpu_trunc_norm.cu
// Author: Nick Ulle

#include <stdio.h>
#include <math.h>
#include <curand_kernel.h>

#define NUM_RNG 128
#define MAX_ITER 100

__device__ float tail_norm(
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

__device__ int get_thread_id()
{
    // Get block and thread numbers.
    int block = blockIdx.x + blockIdx.y * gridDim.x;
    int thread = threadIdx.x + threadIdx.y * blockDim.x
        + threadIdx.z * (blockDim.y * blockDim.x);
    int block_size = blockDim.x * blockDim.y * blockDim.z;
    return thread + block * block_size;
}

extern "C" {
__global__ void gpu_random_activate(int seed, curandState_t **state)
{
    int idx = get_thread_id();

    if (idx < NUM_RNG)
    {
        curandState_t *rng_state = \
            (curandState_t*) malloc(sizeof(curandState_t));
        curand_init(seed, idx, 0, rng_state);
        state[idx] = rng_state;
    }
}

__global__ void gpu_random_deactivate(curandState_t **state)
{
    int idx = get_thread_id();

    if (idx < NUM_RNG) free(state[idx]);
}

__global__ void gpu_trunc_norm(
    int n, 
    float *mean, 
    float *sd,
    float *a,
    float *b,
    float *result,
    curandState_t **state)
{
    int idx = get_thread_id();

    // Only compute if the index is less than the result length.
    if (idx < n)
    {
        // Determine which RNG to use.
        curandState_t *rng_state = state[idx % NUM_RNG];

        // Get truncated normal value. Use C. Robert's algorithm if possible.
        float draw = 0;
        if (!isfinite(a[idx]) && b[idx] <= mean[idx]) 
        {
            draw = tail_norm(mean[idx], sd[idx], b[idx], rng_state);
        } else if (!isfinite(b[idx]) && a[idx] >= mean[idx]) 
        {
            /*
            printf("RT %i has a = %f, b = %f, mean = %f\n", idx, a[idx], \
                   b[idx], mean[idx]);
            */
            draw = tail_norm(mean[idx], sd[idx], a[idx], rng_state);
        } else 
        {
            /*
            printf("N %i has a = %f, b = %f, mean = %f\n", idx, a[idx], \
                   b[idx], mean[idx]);
            */
            for (int i = 0; i < MAX_ITER; i++)
            {
                draw = sd[idx] * curand_normal(rng_state) + mean[idx];

                if (a[idx] <= draw && draw <= b[idx]) break;
            }
        } // end if
        
        result[idx] = draw;
    }
}
} // end extern

