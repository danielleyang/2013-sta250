/* gpu_trunc_norm.cu
 * Author: Nick Ulle
 * Description:
 *  CUDA C functions for generating truncated normal random variables.
 *
 *  Compile with:
 *      nvcc --ptx -arch=compute_20 gpu_trunc_norm.cu -o gpu_trunc_norm.ptx
 */

#include <stdio.h>
#include <math.h>
#include <curand_kernel.h>

#define NUM_RNG 128
// Set maximum number of iterations for all rejection sampling loops.
#define MAX_ITER 100

__device__ float one_sided_norm(
    float mean, 
    float sd, 
    float a, 
    curandState_t *rng_state)
    /* Sample a random value from a one-sided normal distribution.
     *
     * Args:
     *  mean: the mean.
     *  sd: the standard deviation.
     *  a: the finite truncation point.
     *  rng_state: the random number generator to be used.
     */
{
    a = (a - mean) / sd;
    // The algorithm samples from the right tail of N(0, 1); mirror everything
    // if left tail samples are requested.
    float mirror = 1;
    if (a < 0) mirror = -1;
    a *= mirror;

    float z = 0;
    for (int i = 0; i < MAX_ITER; i++)
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
    return sd * mirror * z + mean;
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
    __global__ void gpu_curand_init(int seed, curandState_t **state)
        /* Initialize random number generators.
         *
         * Args:
         *  seed: a seed value.
         *  state: pointer to the random number generators.
         */
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

    __global__ void gpu_curand_deinit(curandState_t **state)
        /* De-initialize random number generators.
         *
         * Args:
         *  state: pointer to the random number generators.
         */
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
        /* Sample random values from a truncated normal distribution.
         *
         * Args:
         *  n: number of values to sample.
         *  mean: array of means.
         *  sd: array of standard deviations.
         *  a: array of lower truncation points.
         *  b: array of upper truncation points.
         *  result: array to store the random values.
         *  state: pointer to the random number generators.
         */
    {
        int idx = get_thread_id();

        // Only compute if the index is less than the result length.
        if (idx < n)
        {
            // Choose RNG state based on thread ID within block. A better
            // solution would be to run only one block and have each thread
            // generate many random values.
            curandState_t *rng_state = state[idx % NUM_RNG];

            // Draw a truncated normal value using vanilla rejection sampling
            // if the truncation region includes the mean; otherwise, use the
            // one-sided algorithm described in Robert (2009).
            float draw = 0;
            if (!isfinite(a[idx]) && b[idx] <= mean[idx]) 
            { // Use one-sided algorithm.
                draw = one_sided_norm(mean[idx], sd[idx], b[idx], rng_state);
            } else if (!isfinite(b[idx]) && a[idx] >= mean[idx]) 
            {
                draw = one_sided_norm(mean[idx], sd[idx], a[idx], rng_state);
            } else 
            { // Use vanilla rejection sampling.
                for (int i = 0; i < MAX_ITER; i++)
                {
                    draw = sd[idx] * curand_normal(rng_state) + mean[idx];

                    if (a[idx] <= draw && draw <= b[idx]) break;
                }
            } // end if
        
            result[idx] = draw;
        } // end if
    }

} // end extern

