from functools import wraps
import numpy as np
from statistical_parts.math_parts.wmw_exact_power import min_are_quantile
import cython
import sys

# TO DO: document this
sys.setrecursionlimit(2000)


@cython.locals(ks=cython.int[:], m=cython.int, n=cython.int)
def vect_MW_CDF(ks, m, n):
    return np.array([fixed_MW_CDF(k, m, n) for k in ks])


@cython.locals(k=cython.int, m=cython.int, n=cython.int)
def fixed_MW_CDF(k, m, n):
    """ Returns the CDF in the MW test of getting U = k with sample sizes m and n"""
    i: cython.int

    if k == m*n/2:
        return 0.5
    elif k < m*n/2:
        result = 0
        t = comb(m+n, n)
        for i in range(k + 1):
            result += n_partition(i, m, n)/t
    else:
        result = 1
        t = comb(m + n, n)
        for i in range(m*n - k):
            result -= n_partition(i, m, n) / t

    return result


def cached(func):
    func.cache = {}

    @wraps(func)
    def wrapper(*args):
        try:
            # return value if run before
            return func.cache[f'{args}']
        except KeyError:
            # run function, store results in func.cache
            result = func(*args)
            if isinstance(result, np.ndarray):
                func.cache[f'{args}'] = tuple(map(tuple, result))
            else:
                func.cache[f'{args}'] = result

            return result
    return wrapper


@cython.locals(k=cython.int, m=cython.int, n=cython.int)
def n_partition(k, m, n):
    """
    Returns number of partitions with m parts of max size n summing to k
    This equals the number of combinations with U statistic = k, if the groups have sample sizes n and m
    """
    if k < 0 or m < 0 or n < 0 or k > m*n:
        return 0

    if k == 0:
        return 1

    if n > m:
        if k > m*n/2:
            return non_redundant_n_partition(m*n - k, n, m)
        else:
            return non_redundant_n_partition(k, n, m)
    elif k > m*n/2:
        return non_redundant_n_partition(m*n-k, m, n)

    return non_redundant_n_partition(k, m, n)


@cached
def non_redundant_n_partition(k, m, n):
    """ Written in order not to cache all redundant options"""
    result = n_partition(k - n, m - 1, n)
    result += n_partition(k, m, n - 1)
    return result


@cython.cfunc
@cython.returns(cython.double)
@cython.locals(n=cython.int, k=cython.int)
@cython.cdivision(True)
def comb(n, k):
    result: cython.double
    i: cython.int

    result = 1

    if k < 0:
        result = 0
    elif k < n - k:
        for i in range(k):
            result *= n - k + i + 1
            result = result / (i + 1)
    else:
        for i in range(n - k):
            result *= k + i + 1
            result = result / (i + 1)

    return result


@cython.cfunc
@cython.returns(cython.int[:, :])
@cython.locals(group1_data=cython.double[:, :], group2_data=cython.double[:, :], ns=cython.int[:, :])
def U_statistics_intern(group1_data, group2_data, ns):
    # c function, cannot be imported to another python script
    n_analyses: cython.int
    n_sims: cython.int
    i: cython.int
    j: cython.int
    k: cython.int
    s: cython.int
    us: cython.int[:, :]

    n_analyses = ns.shape[0]
    n_sims = group1_data.shape[1]

    us = np.zeros((n_analyses, n_sims), dtype=int)

    for j in range(ns[0, 0]):
        for k in range(ns[0, 1]):
            for s in range(n_sims):
                us[0, s] = us[0, s] + int(group1_data[j, s] > group2_data[k, s])

    for i in range(n_analyses-1):
        for j in np.arange(ns[i, 0], ns[i + 1, 0]):
            for k in range(ns[i, 1]):
                for s in range(n_sims):
                    us[i + 1, s] = us[i + 1, s] + int(group1_data[j, s] > group2_data[k, s])

        for j in range(ns[i + 1, 0]):
            for k in np.arange(ns[i, 1], ns[i + 1, 1]):
                for s in range(n_sims):
                    us[i + 1, s] = us[i + 1, s] + int(group1_data[j, s] > group2_data[k, s])

    us = np.cumsum(us, axis=0)
    return us


def U_stats(group1_data, group2_data, ns):
    # python function, can be imported to another python script
    return U_statistics_intern(group1_data, group2_data, ns)


def simulate_U_stats(n, sample_sizes, cohens_d, pdf='Min ARE'):
    group1_data, group2_data = base_simulator(n, sample_sizes, pdf)
    return U_statistics_intern(group1_data + cohens_d, group2_data, sample_sizes)


def base_simulator(n, sample_sizes, pdf):
    if pdf == 'Normal':
        return np.random.normal(size=n * sample_sizes[-1, 0]).reshape(sample_sizes[-1, 0], n), \
            np.random.normal(size=n * sample_sizes[-1, 1]).reshape(sample_sizes[-1, 1], n)
    elif pdf == 'Uniform':
        return np.random.uniform(size=n * sample_sizes[-1, 0]).reshape(sample_sizes[-1, 0], n), \
               np.random.uniform(size=n * sample_sizes[-1, 1]).reshape(sample_sizes[-1, 1], n)
    elif pdf == 'Min ARE':
        return min_are_quantile(np.random.uniform(size=n * sample_sizes[-1, 0]).reshape(sample_sizes[-1, 0], n)), \
               min_are_quantile(np.random.uniform(size=n * sample_sizes[-1, 1]).reshape(sample_sizes[-1, 1], n))
    else:
        raise ValueError('pdf base simulator not implemented')

