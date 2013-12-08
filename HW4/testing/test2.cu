// test2.cu
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
    return sd * tail * z + mean;
}

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
        float draw;
        if (!isfinite(a[idx]) && b[idx] <= mean[idx]) 
	{
            // printf("Left tail code in thread %i.\n", idx);
            draw = tail_normal(mean[idx], sd[idx], b[idx], &rng_state);
        } else if (!isfinite(b[idx]) && a[idx] >= mean[idx]) 
        {
            // printf("Right tail code in thread %i.\n", idx);
            draw = tail_normal(mean[idx], sd[idx], a[idx], &rng_state);
        } else 
        {
            // printf("Generic code in thread %i.\n", idx);
            for (int i = 0; i < MAX_ITER; i++)
            {
                draw = sd[idx] * curand_normal(&rng_state) + mean[idx];

                if (a[idx] <= draw && draw <= b[idx]) break;
            }
        } // end if

        result[idx] = draw;
    }
}

int main()
{
    int n = 3; // desired number of samples
    float mean[] = {0, 0, 0};
    float sd[] = {0, 1, 1};
    float a[] = {-INFINITY, 20, -INFINITY};
    float b[] = {INFINITY, INFINITY, -20};
    float res[n];

    // Initialize GPU memory.
    float *d_mean, *d_sd, *d_a, *d_b, *d_res;
    cudaMalloc(&d_mean, sizeof(mean));
    cudaMalloc(&d_sd, sizeof(sd));
    cudaMalloc(&d_a, sizeof(a));
    cudaMalloc(&d_b, sizeof(b));
    cudaMalloc(&d_res, sizeof(res));

    // Copy to GPU memory.
    cudaMemcpy(d_mean, mean, sizeof(mean), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sd, sd, sizeof(sd), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a, a, sizeof(a), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(b), cudaMemcpyHostToDevice);

    // Run the kernel.
    gpu_trunc_normal<<<1, 16>>>(n, d_mean, d_sd, d_a, d_b, d_res);

    cudaDeviceSynchronize();
    // Copy back to host memory.
    printf("Size of res is %zu\n", sizeof(res));
    cudaMemcpy(res, d_res, sizeof(res), cudaMemcpyDeviceToHost);

    // Print the result.
    puts("The results are in!");
    for (int i = 0; i < n; i++)
    {
        printf(" %f", res[i]);
        if (i == 4)
        {
            puts("");
        }
    }
    puts("");
}

