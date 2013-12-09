/* cpu_trunc_norm.cpp
 * Author: Nick Ulle
 * Description:
 *  Native C/C++ functions for generating truncated normal random variables,
 *  using the new random library added in the C++11 standard.
 *
 *  Compile with:
 *      g++ -std=c++0x -fPIC -shared -o cpu_trunc_norm.so cpu_trunc_norm.cpp
 */

#include <random>
#include <iostream>
#include <cmath>
using namespace std;

// Set maximum number of iterations for all rejection sampling loops.
#define MAX_ITER 100

// Use the 32-bit Mersenne Twister as the RNG.
mt19937 rng_state;
// Set up functions for standard uniform and normal sampling.
normal_distribution<float> rnorm(0, 1);
uniform_real_distribution<float> runif(0, 1);

float one_sided_norm(float mean, float sd, float a)
    /* Sample a random value from a one-sided normal distribution.
     *
     * Args:
     *  mean: the mean.
     *  sd: the standard deviation.
     *  a: the finite truncation point.
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
        float u = runif(rng_state);
        z = -logf(1 - u) / alpha + a;

        // Compute g(z).
        float gz = expf(-powf(z - alpha, 2) / 2);

        // Generate u and test acceptance.
        u = runif(rng_state);

        if (u <= gz) break;
    }
    return sd * mirror * z + mean;
}

extern "C"
{
    void set_seed(int seed)
        /* Set the seed for random number generation.
         *
         * Args:
         *  seed: a seed value.
         */
    {
        rng_state = mt19937(seed);
    }

    void cpu_trunc_norm(
        int n, 
        float *mean, 
        float *sd, 
        float *a, 
        float *b, 
        float *result)
        /* Sample random values from a truncated normal distribution.
         *
         * Args:
         *  n: number of values to sample.
         *  mean: array of means.
         *  sd: array of standard deviations.
         *  a: array of lower truncation points.
         *  b: array of upper truncation points.
         *  result: array to store the random values.
         */
    {
        for (int idx = 0; idx < n; idx++)
        {
            // Draw a truncated normal value using vanilla rejection sampling
            // if the truncation region includes the mean; otherwise, use the
            // one-sided algorithm described in Robert (2009).
            float draw = 0;
            if (isinf(a[idx]) && b[idx] <= mean[idx]) 
            { // Use one-sided algorithm.
                draw = one_sided_norm(mean[idx], sd[idx], b[idx]);
            } else if (isinf(b[idx]) && a[idx] >= mean[idx]) 
            {
                draw = one_sided_norm(mean[idx], sd[idx], a[idx]);
            } else 
            { // Use vanilla rejection sampling.
                for (int i = 0; i < MAX_ITER; i++)
                {
                    draw = sd[idx] * rnorm(rng_state) + mean[idx];

                    if (a[idx] <= draw && draw <= b[idx]) break;
                }
            } // end if
            
            result[idx] = draw;
        } // end for
    }

} // end extern

/*
int main()
{
    int n = 5;
    float mean[5] = { };
    float sd[5] = {1, 1, 1, 1, 1};
    float a[5] = {10, 10, 10, 10, 10};
    float b[5] = {INFINITY, INFINITY, INFINITY, INFINITY, INFINITY};
    float result[5] = { };

    set_seed(10);
    cpu_trunc_norm(n, mean, sd, a, b, result);

    for (int i = 0; i < n; i++)
    {
        printf(" %f", result[i]);
    }
    printf("\n");
}
*/

