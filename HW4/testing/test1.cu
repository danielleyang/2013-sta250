
#include <stdio.h>

// Define a kernel.
__global__ void vecAdd(float *A, float *B, float *C, int N)
{
    // Get the ID of the thread.
	int i = threadIdx.x;
    // If the thread corresponds to an element of the vector, perform the
    // summation.
    if (i < N)
    {
        C[i] = A[i] + B[i];
    }
}

__global__ void vecCos(float *A, int N)
{
    int i = threadIdx.x;
    if (i < N)
    {
        A[i] = normcdff(A[i]);
    }
}

__global__ void vecRand(float *A, int N)
{
    int i = threadIdx.x;
    if (i < N)
    {
        curandState_t rng_state;
        curand_init(5151, i, 0, &rng_state);
        A[i] = curand_normal(&rng_state);
    }
}

int main()
{
    #define N 3
    #define NUM_BLOCKS 1
    #define NUM_THREADS 4

    float A[N] = {1, 2, 3};
    float B[N] = {3, 2, 1};
    float C[N];

    size_t size = sizeof(A);
    printf("The size of A is %zu.\n", size);

    // Allocate device memory.
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Copy from host memory to device memory.
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    // Run the kernels.
    vecAdd<<<NUM_BLOCKS, NUM_THREADS>>>(d_A, d_B, d_C, N);
    vecCos<<<NUM_BLOCKS, NUM_THREADS>>>(d_C, N);
    vecRand<<<NUM_BLOCKS, NUM_THREADS>>>(d_C, N);

    // Copy from device memory to host memory.
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    // Print out the result.
    for (int i = 0; i < N; i++)
    {
        printf("Element %i is %f\n", i, C[i]);
    }
}

