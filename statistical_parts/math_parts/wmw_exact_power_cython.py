import cython
import numpy as np


# region base functions
@cython.wraparound(False)
@cython.boundscheck(False)
def np_shift_min_mat(V, midpoint_vals, n_cols: cython.int, max_rows: cython.int):
    try:
        n_rows = V.shape[0]
    except AttributeError:
        return V

    new_rows = min(max_rows, n_rows + 1)
    new_V = np.zeros((new_rows, n_cols))

    new_V[0, 1:] = np.cumsum(np.sum(V[:, :n_cols-1], axis=0)) * midpoint_vals[1:]
    new_V[1:, :] = V[:new_rows-1, :] * midpoint_vals[np.newaxis, :] / np.arange(2, new_rows + 1)[:, np.newaxis]

    return new_V


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.locals(n=cython.int, m=cython.int, max_rows=cython.int)
def simple_result(n, m, fs, gs, max_rows):
    i: cython.int
    n_cols = fs.size

    if n >= 1:
        V = n * np.copy(fs).reshape((1, n_cols))
    elif m >= 1:
        V = m * np.copy(gs).reshape((1, n_cols))
        m -= 1
    else:
        return np.ones((1, n_cols))

    for i in range(n - 1):
        V = (n - i - 1) * np_shift_min_mat(V, fs, n_cols, max_rows)
    for i in range(m):
        V = (m - i) * np_shift_min_mat(V, gs, n_cols, max_rows)

    return V


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
@cython.returns(cython.double)
@cython.locals(n=cython.int, f=cython.double)
def c_pow(f, n):
    # ONLY FOR POSITIVE INTEGER EXPONENTS
    i: cython.int
    j: cython.int
    result: cython.double
    if n == 0:
        return 1

    result = f
    for j in range(n-1):
        result *= f
    return result


@cython.cfunc
@cython.returns(cython.double[:, :])
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.locals(n=cython.int, m=cython.int, f_view=cython.double[:], g_view=cython.double[:], max_rows=cython.int)
def c_all_combs_mat(n, m, f_view, g_view, max_rows):
    i: cython.int
    j: cython.int
    a: cython.int
    b: cython.int
    c: cython.int
    d: cython.double
    n_cols: cython.int
    V_view: cython.double[:, :]
    f_c_view: cython.double[:]
    g_c_view: cython.double[:]

    n_cols = len(f_view)
    V = np.zeros((min(max_rows, n+m), n_cols))
    V_view = V

    f_cum = np.cumsum(f_view)[:(n_cols-1)].astype(float)
    g_cum = np.cumsum(g_view)[:(n_cols-1)].astype(float)
    f_c_view = f_cum
    g_c_view = g_cum

    for i in range(min(max_rows, n+m-1)):
        if i+1 > n:
            a = n
            b = i + 1 - n
            c = min(n + 1, m - b + 1)
        else:
            a = i+1
            b = 0
            if m < i + 1:
                c = m + 1
            else:
                c = i + 2

        for _ in range(c):
            d = comb(n, a) * comb(m, b)
            for j in range(n_cols - 1):
                V_view[i, j + 1] = d * c_pow(f_view[j + 1], a) * c_pow(g_view[j + 1], b) * c_pow(f_c_view[j], n-a) * \
                               c_pow(g_c_view[j], m-b) + V_view[i, j + 1]
            a -= 1
            b += 1

    if max_rows >= n + m:
        for j in range(n_cols):
            V_view[n+m - 1, j] = c_pow(f_view[j], n) * c_pow(g_view[j], m)

    return V
# endregion


@cython.wraparound(False)
@cython.boundscheck(False)
def HA_CDF_matrix(ks_start, ns_start, ms_start, fs, gs, h, max_rows: cython.int):
    i: cython.int
    cache = {}
    n_cols = fs.size
    results = []

    def check_redundancy(ks, ns, ms, n_analyses: cython.int):
        i_1: cython.int
        j: cython.int
        k_view: cython.int[:]
        n_view: cython.int[:]
        m_view: cython.int[:]

        redundant = np.zeros(n_analyses, dtype=bool)
        ks_copy = np.copy(ks)
        k_view = ks_copy
        n_view = ns
        m_view = ms

        if k_view[0] < 0 or n_view[0] < 0 or m_view[0] < 0:
            return 0

        redundant[0] = k_view[0] >= n_view[0] * m_view[0]

        for i_1 in range(n_analyses - 1):
            i_1 += 1
            if k_view[i_1] < 0 or n_view[i_1] < 0 or m_view[i_1] < 0 or n_view[i_1] < n_view[i_1 - 1] or \
                    m_view[i_1] < m_view[i_1 - 1]:
                return 0

            if n_view[i_1] * m_view[i_1] - n_view[i_1-1] * m_view[i_1-1] + k_view[i_1-1] <= k_view[i_1]:
                k_view[i_1] = n_view[i_1] * m_view[i_1] - n_view[i_1-1] * m_view[i_1-1] + k_view[i_1-1]
                redundant[i_1] = True
            elif k_view[i_1] <= k_view[i_1-1]:
                redundant[i_1-1] = True

        if np.all(redundant):
            return c_all_combs_mat(n_view[n_analyses-1], m_view[n_analyses-1], fs, gs, max_rows)

        redundant[n_analyses-1] = False

        return cache_and_calc(ks_copy[np.logical_not(redundant)], ns[np.logical_not(redundant)],
                              ms[np.logical_not(redundant)], n_analyses - np.sum(redundant))

    def cache_and_calc(ks, ns, ms, n_analyses):
        try:
            # return value if run before
            return cache[f'{ns}'][f'{ms}'][f'{ks}']
        except KeyError:
            # run function, store results in func.cache
            result = non_redundant(ks, ns, ms, n_analyses)

            try:
                cache[f'{ns}'][f'{ms}'][f'{ks}'] = result
            except KeyError:
                try:
                    cache[f'{ns}'][f'{ms}'] = {f'{ks}': result}
                except KeyError:
                    cache[f'{ns}'] = {f'{ms}': {f'{ks}': result}}
            return result

    def non_redundant(ks, ns, ms, n_analyses):
        i_1: cython.int
        simple: cython.int

        k_view: cython.int[:]
        n_view: cython.int[:]
        m_view: cython.int[:]
        k_view = ks
        n_view = ns
        m_view = ms

        simple = 1
        for i_1 in range(n_analyses):
            if k_view[i_1] != 0:
                simple = 0
                break

        result = simple_result(n_view[n_analyses - 1], m_view[n_analyses - 1], fs, gs, max_rows)
        if simple:
            return result

        temp_ks = np.copy(ks)
        temp_ns = np.copy(ns)
        temp_ms = np.copy(ms)

        m_rel: cython.int
        n_rel: cython.int
        m_rel = 1
        n_rel = 1

        for i_1 in range(n_analyses):
            if n_rel:
                if temp_ns[n_analyses - 1 - i_1] > 0 and \
                        (temp_ks[n_analyses - 1 - i_1] >= m_view[n_analyses - 1 - i_1]):
                    temp_ns[n_analyses - 1 - i_1] -= 1
                    temp_ks[n_analyses - 1 - i_1] -= m_view[n_analyses - 1 - i_1]

                    if i_1 == n_analyses - 1:
                        result += ns[0] * np_shift_min_mat(check_redundancy(temp_ks, temp_ns, ms, n_analyses),
                                                           fs, n_cols, max_rows)
                    elif temp_ns[n_analyses - 1 - i_1] >= temp_ns[n_analyses - 2 - i_1]:
                        result += (ns[n_analyses - 1 - i_1] - ns[n_analyses - 2 - i_1]) * \
                                  np_shift_min_mat(check_redundancy(temp_ks, temp_ns, ms, n_analyses), fs, n_cols,
                                                   max_rows)
                else:
                    n_rel = 0

            if m_rel:
                if temp_ms[n_analyses - 1 - i_1] > 0:
                    temp_ms[n_analyses - 1 - i_1] -= 1

                    if i_1 == n_analyses - 1:
                        result += ms[0] * np_shift_min_mat(check_redundancy(ks, ns, temp_ms, n_analyses), gs, n_cols,
                                                           max_rows)
                        result -= ms[0] * np_shift_min_mat(simple_result(ns[n_analyses - 1], temp_ms[n_analyses - 1],
                                                                         fs, gs, max_rows), gs, n_cols, max_rows)

                    elif temp_ms[n_analyses - 1 - i_1] >= temp_ms[n_analyses - 2 - i_1]:
                        result += (ms[n_analyses - 1 - i_1] - ms[n_analyses - 2 - i_1]) * \
                            np_shift_min_mat(check_redundancy(ks, ns, temp_ms, n_analyses), gs, n_cols, max_rows)

                        result -= (ms[n_analyses - 1 - i_1] - ms[n_analyses - 2 - i_1]) * \
                            np_shift_min_mat(simple_result(ns[n_analyses - 1], temp_ms[n_analyses - 1], fs, gs,
                                                           max_rows), gs, n_cols, max_rows)
                else:
                    m_rel = 0
        return result

    if type(ks_start) == np.ndarray and len(ks_start.shape) == 1:
        meaningful_analyses = np.asarray(ks_start < ns_start * ms_start)
        if np.any(meaningful_analyses):
            a = ks_start[meaningful_analyses]
            meaningful_analyses[meaningful_analyses] = np.append(a[1:] > a[:len(a) - 1], True)
        if np.any(meaningful_analyses):
            a = ks_start[meaningful_analyses]
            b = ns_start[meaningful_analyses]
            c = ms_start[meaningful_analyses]
            meaningful_analyses[meaningful_analyses] = np.append(True, a[1:] < a[:len(a) - 1] + b[1:] * c[1:] -
                                                                 b[:len(b) - 1] * c[:len(c) - 1])
        if not np.any(meaningful_analyses):
            return 1

        last_rel_analysis = np.max(np.arange(ns_start.shape[0])[meaningful_analyses])
        return np.sum(check_redundancy(ks_start[meaningful_analyses], ns_start[meaningful_analyses],
                                       ms_start[meaningful_analyses], np.sum(meaningful_analyses))) * \
            h ** (ns_start[last_rel_analysis] + ms_start[last_rel_analysis])
    elif type(ks_start) == np.ndarray:
        for i in range(ks_start.shape[0]):
            meaningful_analyses = np.asarray(ks_start[i, :] < ns_start * ms_start)
            if np.any(meaningful_analyses):
                a = ks_start[i, meaningful_analyses]
                meaningful_analyses[meaningful_analyses] = np.append(a[1:] > a[:len(a)-1], True)
            if np.any(meaningful_analyses):
                a = ks_start[i, meaningful_analyses]
                b = ns_start[meaningful_analyses]
                c = ms_start[meaningful_analyses]
                meaningful_analyses[meaningful_analyses] = np.append(True, a[1:] < a[:len(a) - 1] + b[1:] * c[1:] -
                                                                     b[:len(b)-1] * c[:len(c) - 1])
            if not np.any(meaningful_analyses):
                results.append(1)
            else:
                last_rel_analysis = np.max(np.arange(ns_start.shape[0])[meaningful_analyses])
                results.append(np.sum(check_redundancy(ks_start[i, meaningful_analyses], ns_start[meaningful_analyses],
                                      ms_start[meaningful_analyses], np.sum(meaningful_analyses))) *
                               h ** (ns_start[last_rel_analysis] + ms_start[last_rel_analysis]))
    elif hasattr(ks_start, '__iter__'):
        for these_ks in ks_start:
            try:
                meaningful_analyses = np.asarray(these_ks < ns_start * ms_start)
                if np.any(meaningful_analyses):
                    a = these_ks[meaningful_analyses]
                    meaningful_analyses[meaningful_analyses] = np.append(a[1:] > a[:len(a) - 1], True)
                if np.any(meaningful_analyses):
                    a = these_ks[meaningful_analyses]
                    b = ns_start[meaningful_analyses]
                    c = ms_start[meaningful_analyses]
                    meaningful_analyses[meaningful_analyses] = np.append(True, a[1:] < a[:len(a) - 1] + b[1:] * c[1:] -
                                                                         b[:len(b) - 1] * c[:len(c) - 1])
                if not np.any(meaningful_analyses):
                    results.append(1)
                else:
                    last_rel_analysis = np.max(np.arange(ns_start.shape[0])[meaningful_analyses])
                    results.append(np.sum(check_redundancy(these_ks[meaningful_analyses], ns_start[meaningful_analyses],
                                          ms_start[meaningful_analyses], np.sum(meaningful_analyses))) *
                                   h ** (ns_start[last_rel_analysis] + ms_start[last_rel_analysis]))
            except ValueError:
                print(f'{these_ks}' + " in ks_start have a different number of analyses than the sample sizes")
            except TypeError:
                print(f'{these_ks}' + " in ks_start has the wrong type")
        else:
            raise TypeError("ks_start is not an accepted type")
    return results

