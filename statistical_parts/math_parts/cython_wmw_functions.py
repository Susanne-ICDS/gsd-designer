from functools import wraps
import json
import numpy as np
import os
import cython
import numbers
import sys

sys.setrecursionlimit(2000)


@cython.locals(k=cython.int, m=cython.int, n=cython.int)
def fixed_MW_CDF(k, m, n):
    """ Returns the CDF in the MW test of getting U = k with sample sizes m and n"""
    i: cython.int

    if k == m*n/2:
        return 0.5
    elif k < m*n/2:
        result = 0
        t = comb(m+n, n)
        for i in range(k):
            result += n_partition(i, m, n)/t
    else:
        result = 1
        t = comb(m + n, n)
        for i in range(m*n - k):
            result -= n_partition(i, m, n) / t

    return result


def cached(func):
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




