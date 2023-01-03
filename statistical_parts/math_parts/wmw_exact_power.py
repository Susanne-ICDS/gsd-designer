import numpy as np
from statistical_parts.math_parts.wmw_exact_power_cython import HA_CDF_matrix
from scipy.stats import norm
from scipy.optimize import root_scalar
import multiprocessing as mp


def min_are_pdf(x):
    if x*x < 5:
        a = (5 ** 0.5)
        b = 3 / 20 / (5 ** 0.5)
        return b * (a**2 - x**2)
    else:
        return 0


def min_are_quantile(ps):
    qs = 20**0.5 * np.cos(1/3*np.arccos(1-2*ps) + 4*np.pi/3)
    return qs


def clean_input(ks, ns, ms, sig_is_upper):
    if len(ks.shape) == 1:
        if sig_is_upper:
            ks = (ns * ms) - ks
            ms_copy = ms.copy()
            ms = ns.copy()
            ns = ms_copy

        ks[ks > (ns * ms)] = (ns * ms)[ks > (ns * ms)]
        rel_ks = np.all(ks >= 0)
    else:
        if sig_is_upper:
            ks = (ns * ms)[np.newaxis, ] - ks[np.arange(ks.shape[0]-1, -1, -1), ]

            ms_copy = ms.copy()
            ms = ns.copy()
            ns = ms_copy

        ks[ks > (ns * ms)[np.newaxis, ]] = np.tile(ns * ms, [ks.shape[0], 1])[ks > (ns * ms)[np.newaxis, ]]
        rel_ks = np.all(ks >= 0, axis=1)

    return ks, ns, ms, rel_ks


def HA_all_sig_approximation(ks, ns, ms, shift, pdf, combinations=None, int_range=None, max_rows=6, tol=10 ** -5,
                             max_iter=11, max_richardson=10, manager=None, max_parallel=0, return_intermediate=False,
                             sig_is_upper=False):

    ks, ns, ms, rel_ks = clean_input(ks, ns, ms, sig_is_upper)
    a = np.zeros(ks.shape[0]).tolist()
    if len(ks.shape) == 1:
        n_ks = 1
        if not rel_ks:
            if return_intermediate:
                return [0, 0]
            return 0

    else:
        n_ks = ks.shape[0]
        if not np.any(rel_ks):
            if return_intermediate:
                return [a, a]
            return a

    a = np.zeros(n_ks).tolist()

    if combinations is not None:
        rel_combos = np.array([np.any(combo * rel_ks) for combo in combinations])
        combos = []
        for i, combo in enumerate(combinations):
            if rel_combos[i]:
                combos.append(combo[rel_ks])
    else:
        combos = None

    if return_intermediate:
        intermediates, results = \
            HA_iteration(ks[rel_ks], ns, ms, shift, pdf, int_range, max_rows, tol, max_iter, max_richardson,
                         manager, max_parallel, return_intermediate, tol_type='absolute', combinations=combos)
        if n_ks == 1:
            return results, intermediates

        for i, j in enumerate(np.arange(0, n_ks, 1)[rel_ks]):
            if sig_is_upper:
                a[-j] = results[i]
            else:
                a[j] = results[i]

        return a, intermediates
    results = HA_iteration(ks[rel_ks], ns, ms, shift, pdf, int_range, max_rows, tol, max_iter, max_richardson,
                           manager, max_parallel, return_intermediate, tol_type='absolute', combinations=combos)
    if n_ks == 1:
        return results
    for i, j in enumerate(np.arange(0, n_ks, 1)[rel_ks]):
        if sig_is_upper:
            a[-j] = results[i]
        else:
            a[j] = results[i]

    return a


def check_power(ks, ns, ms, shift, pdf, required_power, combinations=None, int_range=None, max_rows=6, tol=10**-5,
                max_iter=11, max_richardson=10, manager=None, max_parallel=0, return_intermediate=False,
                solution="underpowered", sig_is_upper=False):
    ks, ns, ms, rel_ks = clean_input(ks, ns, ms, sig_is_upper)
    a = np.zeros(ks.shape[0]).tolist()
    if len(ks.shape) == 1:
        n_ks = 1
        if not rel_ks:
            if return_intermediate:
                return [0, 0]
            return 0

    else:
        n_ks = ks.shape[0]
        if not np.any(rel_ks):
            if return_intermediate:
                return [a, a]
            return a

    a = np.zeros(n_ks).tolist()

    if combinations is not None:
        rel_combos = np.array([np.any(combo * rel_ks) for combo in combinations])
        combos = []
        for i, combo in enumerate(combinations):
            if rel_combos[i]:
                combos.append(combo[rel_ks])
        ref_value = required_power[rel_combos]
    else:
        ref_value = required_power
        combos = None

    if return_intermediate:
        intermediates, results = \
            HA_iteration(ks[rel_ks], ns, ms, shift, pdf, int_range, max_rows, tol, max_iter, max_richardson,
                         manager, max_parallel, return_intermediate, tol_type='relative',
                         ref_value=ref_value, combinations=combos, solution=solution)

        for i, j in enumerate(np.arange(0, n_ks, 1)[rel_ks]):
            if sig_is_upper:
                a[-j] = results[i]
            else:
                a[j] = results[i]

        return intermediates, a

    results = HA_iteration(ks[rel_ks], ns, ms, shift, pdf, int_range, max_rows, tol, max_iter, max_richardson, manager,
                           max_parallel, return_intermediate, tol_type='relative', ref_value=ref_value,
                           combinations=combos, solution=solution)

    for i, j in enumerate(np.arange(0, n_ks, 1)[rel_ks]):
        if sig_is_upper:
            a[-j] = results[i]
        else:
            a[j] = results[i]
    return a


def HA_CDF_approximation(ks, ns, ms, shift, pdf, combinations=None, int_range=None, max_rows=6, tol=10**-5,
                         max_iter=11, max_richardson=10, manager=None, max_parallel=0, return_intermediate=False):
    return HA_all_sig_approximation(ks, ms, ns, -shift, pdf, combinations, int_range, max_rows, tol, max_iter,
                                    max_richardson, manager, max_parallel, return_intermediate, sig_is_upper=False)


def check_TypeII(ks, ns, ms, shift, pdf, ref_val, combinations=None, int_range=None, max_rows=6, tol=10**-5,
                 max_iter=11, max_richardson=10, manager=None, max_parallel=0, return_intermediate=False,
                 solution="lower"):
    if solution == "lower":
        solution = "underpowered"
    elif solution == "higher":
        solution = "overpowered"
    return check_power(ks, ms, ns, -shift, pdf, ref_val, combinations=combinations, int_range=int_range,
                       max_rows=max_rows, tol=tol, max_iter=max_iter, max_richardson=max_richardson, manager=manager,
                       max_parallel=max_parallel, return_intermediate=return_intermediate, solution=solution,
                       sig_is_upper=False)


# noinspection PyTypeChecker
def HA_iteration(ks, ns, ms, shift, pdf, int_range=None, max_rows=6, tol=10 ** -5, max_iter=15,
                 max_richardson=10, manager=None, max_parallel=0, return_intermediate=False, tol_type='absolute',
                 ref_value=None, combinations=None, solution=None):
    if ref_value is not None and type(ref_value) != np.ndarray:
        ref_value = np.array(ref_value)

    if not (type(ns) == type(ms) == np.ndarray):
        raise TypeError("All input except max_rows should be of type numpy array.")
    if ns.shape != ms.shape or len(ns.shape) != 1:
        raise ValueError("ns_start and ms_start either have different lengths or the wrong dimensions")
    n_analyses = len(ns)
    n = ns[n_analyses - 1]
    m = ms[n_analyses - 1]
    if type(ks) != np.ndarray:
        raise TypeError('ks should be a numpy array')
    if len(ks.shape) == 1:
        ks = ks.reshape((1, ks.size))
    n_ks = ks.shape[0]
    # if the actual range is infinite, tol_multiplier accounts for error due to bounding
    tol_multiplier = 1
    required_borders = None
    if pdf == 'Normal':
        pdf = norm.pdf
        if int_range is None:
            def error_bound_diff(I_0):
                return 1 - (norm.cdf(0.5 * shift + I_0) - norm.cdf(0.5 * shift - I_0)) ** n * \
                       (norm.cdf(-0.5 * shift + I_0) - norm.cdf(-0.5 * shift - I_0)) ** m - tol / 10

            tol_multiplier = 0.9
            try:
                int_bound = root_scalar(error_bound_diff, bracket=[0, 10]).root
            except ValueError:
                int_bound = 10
            int_range = [0.5 * shift - int_bound, 0.5 * shift + int_bound]
    elif pdf == 'Uniform':
        if np.abs(shift) > 1:
            shift = np.sign(shift)
        if shift > 0:
            fs_u = np.array([shift, 1 - shift, 0])
            gs_u = np.array([0, 1 - shift, shift])
        else:
            fs_u = np.array([0, 1 + shift, -shift])
            gs_u = np.array([-shift, 1 + shift, 0])
        return HA_CDF_matrix(ks, ns, ms, fs_u, gs_u, 1, ns[n_analyses - 1] + ms[n_analyses - 1])
    elif pdf == 'Min ARE':
        pdf = min_are_pdf
        if np.abs(shift) > 2 * 5 ** 0.5:
            shift = np.sign(shift) * 2 * 5 ** 0.5
        if shift > 0:
            int_range = [-5 ** 0.5, 5 ** 0.5 + shift]
            required_borders = [-5 ** 0.5 + shift, 5 ** 0.5]
        else:
            int_range = [-5 ** 0.5 + shift, 5 ** 0.5]
            required_borders = [-5 ** 0.5, 5 ** 0.5 + shift]
    elif int_range is None:
        raise ValueError('int_range was not specified')
    interval_length = int_range[1] - int_range[0]
    # make sure there are sufficient interval pieces for the error caused by max_rows to be O(h^2) and also h <= 1
    max_rows = min(n + m, max_rows)
    c_exp = np.ceil(np.log2(max(((n + m) / max_rows), interval_length * 2)))

    def check_done(not_yet_converged):
        still_not_converged = []
        if combinations is not None and tol_type == 'absolute':
            for combo in combinations:
                if np.sum(error_est_results[combo != 0]) > tol_multiplier * tol:
                    still_not_converged += (np.arange(0, n_ks)[combo != 0]).tolist()
        elif combinations is not None and tol_type == 'relative' and solution is None:
            results = np.array([result[len(result) - 1] for result in iteration_results])
            if type(ref_value) == np.ndarray and ref_value.size > 1:
                for i1, combo in enumerate(combinations):
                    if np.abs(np.sum(results * combo) - ref_value[i1]) - np.sum(error_est_results[combo != 0]) < \
                            tol_multiplier * tol < np.sum(error_est_results[combo != 0]):
                        still_not_converged += (np.arange(0, n_ks)[combo != 0]).tolist()
            else:
                for combo in combinations:
                    if np.abs(np.sum(results * combo) - ref_value) - np.sum(error_est_results[combo != 0]) < \
                            tol_multiplier * tol < np.sum(error_est_results[combo != 0]):
                        still_not_converged += (np.arange(0, n_ks)[combo != 0]).tolist()
        elif combinations is not None and tol_type == 'relative' and solution is not None:
            # IMPORTANT: this only works if combinations are ordered from less to more powerful
            results = np.array([result[len(result) - 1] for result in iteration_results])
            if type(ref_value) == np.ndarray and ref_value.size > 1:
                for i1, combo in enumerate(combinations):
                    if np.abs(np.sum(results * combo) - ref_value[i1]) - np.sum(error_est_results[combo != 0]) < \
                            tol_multiplier * tol < np.sum(error_est_results[combo != 0]):
                        still_not_converged += (np.arange(0, n_ks)[combo != 0]).tolist()

                for ref_val in np.unique(ref_value):
                    under = -1
                    rel_combos = []
                    for i2, ref_val2 in enumerate(ref_value):
                        if ref_val2 == ref_val:
                            rel_combos.append(combinations[i2])
                    for (i_1, combo) in enumerate(rel_combos):
                        if np.sum(results * combo) - ref_val <= 0:
                            under += 1
                        else:
                            break
                    if under == -1 and under == len(rel_combos) - 1:
                        pass
                    elif solution == "underpowered":
                        combo = rel_combos[under]
                        if np.sum(error_est_results[combo != 0]) > tol_multiplier * tol:
                            still_not_converged += (np.arange(0, n_ks)[combo != 0]).tolist()
                    else:
                        combo = rel_combos[under + 1]
                        if np.sum(error_est_results[combo != 0]) > tol_multiplier * tol:
                            still_not_converged.append(np.arange(0, n_ks)[combo != 0])

            else:
                for combo in combinations:
                    if np.abs(np.sum(results * combo) - ref_value) - np.sum(error_est_results[combo != 0]) < \
                            tol_multiplier * tol < np.sum(error_est_results[combo != 0]):
                        still_not_converged += (np.arange(0, n_ks)[combo != 0]).tolist()
                under = -1
                for (i_1, combo) in enumerate(combinations):
                    if np.sum(results * combo) - ref_value <= 0:
                        under += 1
                    else:
                        break
                if under == -1 and under == n_ks - 1:
                    pass
                elif solution == "underpowered":
                    combo = combinations[under]
                    if np.sum(error_est_results[combo != 0]) > tol_multiplier * tol:
                        still_not_converged += (np.arange(0, n_ks)[combo != 0]).tolist()
                else:
                    combo = combinations[under + 1]
                    if np.sum(error_est_results[combo != 0]) > tol_multiplier * tol:
                        still_not_converged.append(np.arange(0, n_ks)[combo != 0])
        elif tol_type == 'relative' and solution is None:
            for i_1 in not_yet_converged:
                result = iteration_results[i_1]
                if np.abs(result[len(result) - 1] - ref_value) < tol_multiplier * tol < error_est_results[i_1]:
                    still_not_converged.append(i_1)
        elif tol_type == 'relative' and solution is not None:
            for i_1 in not_yet_converged:
                result = iteration_results[i_1]
                if np.any(np.abs(result[len(result) - 1] - ref_value) < tol_multiplier * tol) and \
                        tol_multiplier * tol < error_est_results[i_1]:
                    still_not_converged.append(i_1)

            for val in ref_value:
                under = -1
                for i_1 in range(n_ks):
                    if iteration_results[i_1][len(iteration_results[i_1]) - 1] < val:
                        under += 1
                    else:
                        break
                if under == -1 or under == n_ks - 1:
                    pass
                elif solution == "underpowered":
                    if tol_multiplier * tol < error_est_results[under]:
                        still_not_converged.append(under)
                else:
                    if tol_multiplier * tol < error_est_results[under + 1]:
                        still_not_converged.append(under + 1)
        elif tol_type == 'absolute':
            for i_1 in not_yet_converged:
                if tol_multiplier * tol < error_est_results[i_1]:
                    still_not_converged.append(i_1)
        still_not_converged = np.unique(still_not_converged).tolist()
        return still_not_converged

    def iterate(outcome_list, ks_now, fs, gs, h):
        outcome_list[:] = HA_CDF_matrix(ks_now, ns, ms, fs, gs, h, max_rows)
        return outcome_list

    def Richardson_expansion(result_list, not_yet_converged, new_elements, error_estimates):
        for i_1, i_2 in enumerate(not_yet_converged):
            old_array = result_list[i_2]
            old_len = len(result_list[i_2])
            new_len = min(old_len + 1, max_richardson)
            new_array = np.zeros(new_len)
            new_array[0] = new_elements[i_1]
            for i_3 in range(new_len - 1):
                new_array[i_3 + 1] = new_array[i_3] + (new_array[i_3] - old_array[i_3]) / (4 ** (i_3 + 1) - 1)
            error_estimates[i_2] = np.abs(new_array[old_len - 1] - old_array[old_len - 1])
            result_list[i_2] = new_array
        return result_list, error_estimates

    # region only relevant when performing the next integration in parallel

    if manager is None:
        max_parallel = 0
    if max_parallel != 0:
        def create_parallel_process(exp, not_yet_converged=None):
            if required_borders is None:
                h_now = interval_length / 2 ** exp
                mid_points = int_range[0] + h_now * (0.5 + np.arange(2 ** exp))
                fs_now = np.array([pdf(point) for point in mid_points])
                gs_now = np.array([pdf(point - shift) for point in mid_points])
            else:
                h_now = np.array([(required_borders[0] - int_range[0]) / 2 ** (exp - 2),
                                 (required_borders[1] - required_borders[0]) / 2 ** (exp - 1)])
                mid_points = np.concatenate([int_range[0] + h_now[0] * (0.5 + np.arange(2 ** (exp - 2))),
                                            required_borders[0] + h_now[1] * (0.5 + np.arange(2 ** (exp - 1))),
                                            required_borders[1] + h_now[0] * (0.5 + np.arange(2 ** (exp - 2)))])
                h_ve = np.ones(int(2 ** c_exp)) * h_now[0]
                h_ve[int(2 ** (c_exp - 2)):int(2 ** (c_exp - 1) + 2 ** (c_exp - 2))] = h_now[1]
                fs_now = np.array([pdf(point) for point in mid_points_0]) * h_ve
                gs_now = np.array([pdf(point - shift) for point in mid_points_0]) * h_ve
                h_now = 1

            p_list = manager.list(range(n_ks))
            if not_yet_converged is None:
                p = mp.Process(target=iterate, args=(p_list, ks, fs_now, gs_now, h_now))
            else:
                p = mp.Process(target=iterate, args=(p_list, ks[not_done, :], fs_now, gs_now, h_now))
            return p, p_list
        list_of_ps = []
        list_of_p_lists = []
        for i in range(max_parallel):
            c_p, c_list = create_parallel_process(c_exp + 1 + i)
            c_p.start()
            list_of_ps.append(c_p)
            list_of_p_lists.append(c_list)
    # endregion

    iteration_results = list(range(n_ks))
    if required_borders is None:
        h_0 = interval_length / 2 ** c_exp
        mid_points_0 = int_range[0] + h_0 * (0.5 + np.arange(2 ** c_exp))
        fs_0 = np.array([pdf(point) for point in mid_points_0])
        gs_0 = np.array([pdf(point - shift) for point in mid_points_0])

    else:
        h_0 = np.array([(required_borders[0]-int_range[0])/2 ** (c_exp-2),
                       (required_borders[1]-required_borders[0])/2 ** (c_exp-1)])
        mid_points_0 = np.concatenate([int_range[0] + h_0[0] * (0.5 + np.arange(2 ** (c_exp-2))),
                                       required_borders[0] + h_0[1] * (0.5 + np.arange(2 ** (c_exp-1))),
                                       required_borders[1] + h_0[0] * (0.5 + np.arange(2 ** (c_exp-2)))])
        h_vec = np.ones(int(2**c_exp)) * h_0[0]
        h_vec[int(2**(c_exp-2)):int(2**(c_exp-1) + 2**(c_exp-2))] = h_0[1]
        fs_0 = np.array([pdf(point) for point in mid_points_0]) * h_vec
        gs_0 = np.array([pdf(point - shift) for point in mid_points_0]) * h_vec
        h_0 = 1

    iteration_results = iterate(iteration_results, ks, fs_0, gs_0, h_0)
    all_results = [iteration_results]
    iteration_results = [np.array([result]) for result in iteration_results]
    not_done = [i for i in range(n_ks)]
    error_est_results = np.zeros(n_ks)
    if max_parallel == 0:
        for _ in range(max_iter):
            c_exp += 1
            if required_borders is None:
                h_0 = interval_length / 2 ** c_exp
                mid_points_0 = int_range[0] + h_0 * (0.5 + np.arange(2 ** c_exp))
                fs_0 = np.array([pdf(point) for point in mid_points_0])
                gs_0 = np.array([pdf(point - shift) for point in mid_points_0])
            else:
                h_0 = np.array([(required_borders[0] - int_range[0]) / 2 ** (c_exp - 2),
                                (required_borders[1] - required_borders[0]) / 2 ** (c_exp - 1)])
                mid_points_0 = np.concatenate([int_range[0] + h_0[0] * (0.5 + np.arange(2 ** (c_exp - 2))),
                                               required_borders[0] + h_0[1] * (0.5 + np.arange(2 ** (c_exp - 1))),
                                               required_borders[1] + h_0[0] * (0.5 + np.arange(2 ** (c_exp - 2)))])
                h_vec = np.ones(int(2 ** c_exp)) * h_0[0]
                h_vec[int(2 ** (c_exp - 2)):int(2 ** (c_exp - 1) + 2 ** (c_exp - 2))] = h_0[1]
                fs_0 = np.array([pdf(point) for point in mid_points_0]) * h_vec
                gs_0 = np.array([pdf(point - shift) for point in mid_points_0]) * h_vec
                h_0 = 1

            new_results = iterate([i for i in not_done], ks[not_done, :], fs_0, gs_0, h_0)
            iteration_results, error_est_results = \
                Richardson_expansion(iteration_results, not_done, new_results, error_est_results)
            not_done = check_done(not_done)

            if return_intermediate:
                all_results.append(new_results)
            if not not_done:
                final = [result[len(result) - 1] for result in iteration_results]
                if return_intermediate:
                    return [all_results, final]
                else:
                    return final
    else:
        # only relevant when performing the next integration in parallel
        list_evaluated = [not_done for _ in range(max_parallel)]
        for i in range(max_iter):
            rel_p = list_of_ps.pop(0)
            new_results = list_of_p_lists.pop(0)
            rel_evaluated = list_evaluated.pop(0)
            rel_p.join()
            if return_intermediate:
                all_results.append(rel_evaluated)
            iteration_results, error_est_results = Richardson_expansion(iteration_results, rel_evaluated,
                                                                        new_results,
                                                                        error_est_results)
            not_done = check_done(rel_evaluated)
            if not not_done:
                for rel_p in list_of_ps:
                    rel_p.terminate()
                    rel_p.join()
                final = [result[len(result) - 1] for result in iteration_results]
                if return_intermediate:
                    return [iteration_results, final]
                return final
            if i <= max_iter - max_parallel:
                c_exp += 1
                c_p, c_list = create_parallel_process(c_exp + max_parallel, not_done)
                list_of_ps.append(c_p)
                list_of_p_lists.append(c_list)
                list_evaluated.append(not_done)
                c_p.start()
    print('Warning: tolerance not reached within max iterations')
    final = [result[len(result) - 1] for result in iteration_results]
    if return_intermediate:
        return [iteration_results, final]
    return final
