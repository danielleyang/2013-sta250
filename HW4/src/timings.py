
def time_sample(k, mean = 2, sd = 1, a = 0, b = 1.5):
    n = 10 ** k
    mean = np.ones(n) * mean
    sd = np.ones(n) * sd
    a = np.ones(n) * a
    b = np.ones(n) * b

    start = cuda.Event()
    end = cuda.Event()

    start.record()
    gpu_trunc_norm(n, mean, sd, a, b)
    end.record()
    end.synchronize()
    gpu_time = start.time_till(end) * 1e-3

    start.record()
    trunc_norm(n, mean, sd, a, b)
    end.record()
    end.synchronize()
    cpu_time = start.time_till(end) * 1e-3

    return gpu_time, cpu_time

def check_sample(n = 10 ** 5, mean = 2, sd = 1, a = 0, b = 1.5):
    n = 10 ** 5
    mean = np.ones(n) * mean
    sd = np.ones(n) * sd
    a = np.ones(n) * a
    b = np.ones(n) * b

    gpu_samp = gpu_trunc_norm(n, mean, sd, a, b)
    cpu_samp = trunc_norm(n, mean, sd, a, b)

    return np.mean(gpu_samp), np.mean(cpu_samp)

def main():
    gpu_random_activate(140)
    np.random.seed(150)

    # Print sample means as diagnostics. These should agree with each other and
    # with the theoretical values.
    print 'N(2, 1) on (0, 1.5) sample means: ', check_sample()
    print 'N(0, 2) on (4, Inf) sample means: ', \
        check_sample(mean = 0, sd = 2, a = 4, b = np.float32('inf'))
    print 'N(-1, 1) on (-Inf, -4) sample means: ', \
        check_sample(mean = -1, sd = 1, a = -np.float32('inf'), b = -4)

    # Run timings.
    time = np.zeros((2, 8))
    for k in range(1, 9):
        time[:, k - 1] = time_sample(k)
        print 'Finished k = {0}.'.format(k)

    np.savetxt('output/timings.csv', time, delimiter = ',')

    gpu_random_deactivate()

# ----- Deprecated
if False:
    _MAX_ITER = 100
    
    def tail_norm(mean, sd, a):
        a = (a - mean) / sd
        tail = -1 if a < 0 else 1
        a *= tail
        for i in range(0, _MAX_ITER):
            # Generate z ~ EXP(alpha) + a.
            alpha = (a + np.sqrt(a**2 + 4)) / 2
            u = np.random.uniform()
            z = -np.log(1 - u) / alpha + a
    
            # Compute g(z).
            gz = np.exp(-(z - alpha)**2 / 2)
    
            u = np.random.uniform()
            if u <= gz:
                break
    
        return sd * tail * z + mean
    
    def trunc_norm(n, mean, sd, a, b):
        result = np.zeros(n)
        for idx in range(0, n):
            if not np.isfinite(a[idx]) and  b[idx] <= mean[idx]:
                draw = tail_norm(mean[idx], sd[idx], b[idx])
            elif not np.isfinite(b[idx]) and a[idx] >= mean[idx]:
                draw = tail_norm(mean[idx], sd[idx], a[idx])
            else:
                for i in range(0, _MAX_ITER):
                    draw = np.random.normal(mean[idx], sd[idx], 1)
                    if a[idx] <= draw and draw <= b[idx]:
                        break
            result[idx] = draw
        return result
