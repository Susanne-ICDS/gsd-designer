import numpy as np
from math import gamma
from scipy.stats import nct, ncf
import statistical_parts.math_parts.t_test_functions as t_gsd
import statistical_parts.math_parts.one_way_functions as f_gsd

from dash import html


"""
n_termination = 2
T = 7.1
sig_bounds = np.array([6.55, 4.8])
fut_bounds = np.array([0, 0])
sample_sizes = np.array([(3, 6), (3, 6), (3, 6)], dtype=int)
alpha = 0.1

test = 'One-way'
one_sided = False
memory_limit = 2
max_iter=10**3
tol_d=10**-2
tol_e=10**-2
"""


def simulate_effect_CI(n_termination, T, sig_bounds, fut_bounds, sample_sizes, alpha, test, one_sided=True,
                       tol_d=10**-3, tol_e=10**-3, memory_limit=2, max_iter=10**3):
    if test == "One-way":
        estimate, lower, higher = simulate_eta2_CI(n_termination, T, sig_bounds, fut_bounds, sample_sizes, alpha,
                                                   tol_d, tol_e, memory_limit, max_iter)
    elif one_sided:
        estimate, lower, higher = simulate_stage_wise_CI(n_termination, T, sig_bounds, fut_bounds, sample_sizes, alpha,
                                                         tol_d, tol_e, memory_limit, max_iter)
    elif np.all(fut_bounds == 0):
        estimate, lower, higher = simulate_stage_wise_CI(n_termination, T, sig_bounds, -sig_bounds, sample_sizes, alpha,
                                                         tol_d, tol_e, memory_limit, max_iter)
    else:
        estimate, lower, higher = simulate_effect_wise_CI(n_termination, T, sig_bounds, fut_bounds, sample_sizes, alpha,
                                                          tol_d, tol_e, memory_limit, max_iter)

    if np.isnan(estimate) or np.isnan(lower) or np.isnan(higher):
        return "The algorithm did not converge. Consider increasing the maximum allowed iterations or the error " \
               "tolerance."

    if n_termination > 1:
        max_round = -int(np.floor(np.log10(tol_d)))

        def rel_round(x):
            if x != 0:
                return round(x, min(-int(np.floor(np.log10(np.abs(x * tol_d)))) - 1, max_round))
            else:
                return x

        estimate = rel_round(estimate)
        lower = rel_round(lower)
        higher = rel_round(higher)

    if test == 'T':
        return ["Cohen's d: {}".format(estimate), html.Br(),
                "{}%-confidence interval: [{}, {}]".format((1 - alpha) * 100, lower, higher)]
    elif test == 'One-way':
        return ["Eta squared: {}".format(estimate), html.Br(),
                "{}%-confidence interval: [{}, {}]".format((1 - alpha) * 100, lower, higher)]


def simulate_stage_wise_CI(n_termination, T, sig_bounds, fut_bounds, sample_sizes, alpha, tol_d, tol_e, memory_limit,
                           max_iter=10**3):
    n1 = sample_sizes[0, n_termination-1]
    n2 = sample_sizes[1, n_termination-1]

    df = n1 + n2 - 2
    nc_parameter = T
    CI_lower = nct.ppf(alpha / 2, df=df, nc=nc_parameter) / (n1 * n2 / (n1 + n2)) ** 0.5
    CI_upper = nct.ppf(1 - alpha / 2, df=df, nc=nc_parameter) / (n1 * n2 / (n1 + n2)) ** 0.5

    if n_termination == 1:
        d_estimate = T * (1 / n1 + 1 / n2) ** 0.5
        return d_estimate, CI_lower, CI_upper

    sample_sizes = sample_sizes[:, :n_termination]
    sig_bounds = sig_bounds[:n_termination]
    fut_bounds = fut_bounds[:n_termination]
    sig_bounds[n_termination - 1] = T
    fut_bounds[n_termination - 1] = T

    def simulator(n, effect_size):
        return t_gsd.simulate_statistics(n, sample_sizes, memory_limit, effect_size, 'one')

    lower, higher = create_solution_interval(CI_upper, simulator, n_termination, sig_bounds, fut_bounds, 1 - alpha/2,
                                             tol_e, max_iter)
    CI_upper = search_interval(lower, higher, simulator, n_termination, sig_bounds, fut_bounds, 1-alpha/2, tol_d, tol_e,
                               max_iter)

    lower, higher = create_solution_interval(CI_lower, simulator, n_termination, sig_bounds, fut_bounds, alpha / 2,
                                             tol_e, max_iter)
    CI_lower = search_interval(lower, higher, simulator, n_termination, sig_bounds, fut_bounds, alpha / 2, tol_d,
                               tol_e, max_iter)

    d_estimate = search_interval(CI_lower, CI_upper, simulator, n_termination, sig_bounds, fut_bounds, 0.5,
                                 tol_d, tol_e, max_iter)

    return d_estimate, CI_lower, CI_upper


def simulate_effect_wise_CI(n_termination, T, sig_bounds, fut_bounds, sample_sizes, alpha, tol_d, tol_e, memory_limit,
                            max_iter=10**3):
    n1 = sample_sizes[0, n_termination - 1]
    n2 = sample_sizes[1, n_termination - 1]
    df = n1 + n2 - 2
    CI_lower = nct.ppf(alpha / 2, df=df, nc=T) / (n1 * n2 / (n1 + n2)) ** 0.5
    CI_upper = nct.ppf(1 - alpha / 2, df=df, nc=T) / (n1 * n2 / (n1 + n2)) ** 0.5

    if n_termination == 1:
        d_estimate = T * (1 / n1 + 1 / n2) ** 0.5
        return d_estimate, CI_lower, CI_upper

    def simulator(n, effect_size):
        return t_gsd.simulate_statistics(n, sample_sizes, memory_limit, effect_size, 'one')

    lower, higher = create_solution_interval(CI_upper, simulator, n_termination, sig_bounds, fut_bounds, 1 - alpha / 2,
                                             tol_e, max_iter, effect_wise=True, T=T, sample_sizes=sample_sizes)
    CI_upper = search_interval(lower, higher, simulator, n_termination, sig_bounds, fut_bounds, 1 - alpha / 2, tol_d,
                               tol_e, max_iter, effect_wise=True, T=T, sample_sizes=sample_sizes)

    lower, higher = create_solution_interval(CI_lower, simulator, n_termination, sig_bounds, fut_bounds, alpha / 2,
                                             tol_e, max_iter, effect_wise=True, T=T, sample_sizes=sample_sizes)
    CI_lower = search_interval(lower, higher, simulator, n_termination, sig_bounds, fut_bounds, alpha / 2, tol_d,
                               tol_e, max_iter, effect_wise=True, T=T, sample_sizes=sample_sizes)

    d_estimate = search_interval(CI_lower, CI_upper, simulator, n_termination, sig_bounds, fut_bounds, 0.5,
                                 tol_d, tol_e, max_iter, effect_wise=True, T=T, sample_sizes=sample_sizes)

    return d_estimate, CI_lower, CI_upper


def simulate_eta2_CI(n_termination, T, sig_bounds, fut_bounds, sample_sizes, alpha, tol_d, tol_e, memory_limit,
                     max_iter=10**3):
    n_groups = sample_sizes.shape[0]
    dfd = np.sum(sample_sizes[:, n_termination - 1]) - n_groups
    eta2 = T/(T + dfd / (n_groups - 1))
    ncp = np.sum(sample_sizes[:, n_termination - 1]) * eta2/(1 - eta2)

    CI_lower = ncf.ppf(alpha / 2, dfn=n_groups - 1, dfd=dfd, nc=ncp)
    CI_lower = CI_lower/(CI_lower + dfd/(n_groups-1))
    CI_upper = ncf.ppf(1-alpha / 2, dfn=n_groups - 1, dfd=dfd, nc=ncp)
    CI_upper = CI_upper / (CI_upper + dfd / (n_groups - 1))

    if n_termination == 1:
        return eta2, CI_lower, CI_upper

    sample_sizes = sample_sizes[:, :n_termination]
    sig_bounds = sig_bounds[:n_termination]
    fut_bounds = fut_bounds[:n_termination]
    sig_bounds[n_termination - 1] = T
    fut_bounds[n_termination - 1] = T

    def simulator(n, eta2_guess):
        lambda_guess = eta2_guess/(1-eta2_guess)
        means = np.zeros(n_groups)
        means[0] = n_groups**0.5 * lambda_guess
        return f_gsd.simulate_statistics(n, sample_sizes, memory_limit, means, 1)

    lower_not_zero = simulate_until_decision(lambda n: simulator(n, 0), n_termination, sig_bounds, fut_bounds, alpha/2,
                                             tol_e, max_iter=max_iter)

    if lower_not_zero == -1:
        lower, higher = create_solution_interval(CI_lower, simulator, n_termination, sig_bounds, fut_bounds, alpha / 2,
                                                 tol_e, max_iter, effect_size_positive=True)
        CI_lower = search_interval(lower, higher, simulator, n_termination, sig_bounds, fut_bounds, alpha / 2, tol_d,
                                   tol_e, max_iter)
    elif lower_not_zero == 0 or lower_not_zero == 1:
        CI_lower = 0
    else:
        CI_lower = np.nan

    lower, higher = create_solution_interval(CI_upper, simulator, n_termination, sig_bounds, fut_bounds, 1 - alpha/2,
                                             tol_e, max_iter, effect_size_positive=True)
    CI_upper = search_interval(lower, higher, simulator, n_termination, sig_bounds, fut_bounds, 1-alpha/2, tol_d, tol_e,
                               max_iter)

    eta2_estimate = search_interval(CI_lower, CI_upper, simulator, n_termination, sig_bounds, fut_bounds, 0.5,
                                    tol_d, tol_e, max_iter)

    return eta2_estimate, CI_lower, CI_upper


def create_solution_interval(start_point, simulator, n_termination, sig_bounds, fut_bounds, val, tol_e, max_iter,
                             effect_size_positive=False, effect_wise=False, T=None, sample_sizes=None):
    higher = lower = np.nan

    if effect_wise:
        flag = simulate_until_decision_effect_wise(lambda n: simulator(n, start_point), n_termination, T, sample_sizes,
                                                   sig_bounds, fut_bounds, val, tol_e, max_iter=max_iter**2)
    else:
        flag = simulate_until_decision(lambda n: simulator(n, start_point), n_termination, sig_bounds, fut_bounds,
                                       val, tol_e, max_iter=max_iter**2)

    if flag == 0:
        return start_point, start_point
    elif flag == 1:
        higher = start_point
        if effect_size_positive:
            new_point = higher * 0.9
        else:
            new_point = higher - max(0.1 * np.abs(higher), 0.1)
    elif flag == -1:
        lower = start_point
        if effect_size_positive:
            new_point = lower * 1.1
        else:
            new_point = lower + max(0.1 * np.abs(lower), 0.1)
    elif np.isnan(flag):
        return np.nan, start_point
    else:
        raise ValueError

    for _ in range(max_iter):
        if effect_wise:
            new_flag = simulate_until_decision_effect_wise(lambda n: simulator(n, new_point), n_termination, T,
                                                           sample_sizes, sig_bounds, fut_bounds, val, tol_e,
                                                           max_iter=max_iter**2)
        else:
            new_flag = simulate_until_decision(lambda n: simulator(n, new_point), n_termination, sig_bounds, fut_bounds,
                                               val, tol_e, max_iter=max_iter**2)

        if new_flag == 0:
            return new_point, new_point
        elif new_flag == 1:
            higher = new_point
            if new_flag == flag:
                if effect_size_positive:
                    new_point = higher * 0.9
                else:
                    new_point = higher - max(0.1 * np.abs(higher), 0.1)
            else:
                break
        elif new_flag == -1:
            lower = new_point
            if new_flag == flag:
                if effect_size_positive:
                    new_point = lower * 1.1
                else:
                    new_point = lower + max(0.1 * np.abs(lower), 0.1)
            else:
                break
        elif np.isnan(new_flag):
            return np.nan, new_point
        else:
            raise ValueError

    return lower, higher


def search_interval(lower, higher, simulator, n_termination, sig_bounds, fut_bounds, val, tol_d, tol_e, max_iter,
                    effect_wise=False, T=None, sample_sizes=None):
    for _ in range(max_iter):
        estimate = 0.5 * (lower + higher)

        if np.abs(lower - higher) < max(np.abs(estimate) * tol_d, tol_d*10**-1):
            return estimate

        if effect_wise:
            flag = simulate_until_decision_effect_wise(lambda n: simulator(n, estimate), n_termination, T, sample_sizes,
                                                       sig_bounds, fut_bounds, val, tol_e, max_iter=max_iter**2)
        else:
            flag = simulate_until_decision(lambda n: simulator(n, estimate), n_termination, sig_bounds, fut_bounds,
                                           val, tol_e, max_iter=max_iter**2)
        if flag == 0:
            return estimate
        elif flag == 1:
            higher = estimate
        elif flag == -1:
            lower = estimate
        elif np.isnan(flag):
            return np.nan
        else:
            raise ValueError(flag)


def simulate_until_decision(simulator, n_termination, sig_bounds, fut_bounds, compare_val, tol_e, base_step=100,
                            max_step=10**6, max_iter=10**6):
    n = max(min(int(max(1/compare_val, 1/(1 - compare_val)) * 100), max_step), base_step)
    n_larger = 0
    n_sims = 0

    decision = np.nan

    for _ in range(max_iter):
        n_sims += n
        new_simulations = simulator(n)
        undecided = np.ones(n, dtype='?')

        for i in range(n_termination):
            larger = new_simulations[i, undecided] > sig_bounds[i]
            smaller = new_simulations[i, undecided] < fut_bounds[i]
            n_larger += np.sum(larger)
            undecided[undecided] = np.logical_and(np.logical_not(smaller), np.logical_not(larger))

        estimate = n_larger/n_sims
        if estimate == 0 or estimate == 1:
            if max(compare_val * (1 - compare_val))**0.5 / (n_sims ** 0.5) * 10 < min(compare_val, 1 - compare_val):
                if estimate == 0:
                    return -1
                else:
                    return 1

            n = max_step
            continue

        sd = (estimate * (1 - estimate))**0.5
        if np.abs(estimate - compare_val) > 2.576 * sd / n_sims ** 0.5:
            decision = np.sign(estimate - compare_val)
            break
        elif 2.576 * sd / n_sims ** 0.5 < max(min(compare_val, 1 - compare_val) * tol_e, tol_e * 10**-1):
            decision = 0
            break
        else:
            if np.abs(estimate - compare_val) == 0:
                n = max_step
            else:
                n = int((2.576 * sd / np.abs(estimate - compare_val)) ** 2 - n_sims) + base_step
                n = max(min(n, max_step), base_step)

    return decision


def simulate_until_decision_effect_wise(simulator, n_termination, T, sample_sizes, sig_bounds, fut_bounds, compare_val,
                                        tol_e, base_step=100, max_step=10**6, max_iter=10**6):
    n = max(min(int(max(1/compare_val, 1/(1 - compare_val)) * 100), max_step), base_step)
    n_larger = 0
    n_sims = 0

    decision = np.nan
    reference_val = T * (np.sum(sample_sizes[:, n_termination - 1])/np.product(sample_sizes[:, n_termination - 1]))**0.5

    for _ in range(max_iter):
        n_sims += n
        new_simulations = simulator(n)
        undecided = np.ones(n, dtype='?')
        stop_at = np.zeros(n, dtype=int)
        for i in range(n_termination-1):
            undecided[undecided] = np.logical_and(np.abs(new_simulations[i, undecided]) < sig_bounds[i],
                                                  np.abs(new_simulations[i, undecided]) > fut_bounds[i])
            stop_at[undecided] += 1

        sim_vals = new_simulations[stop_at, range(n)] * (np.sum(sample_sizes[:, stop_at], axis=0) /
                                                         np.product(sample_sizes[:, stop_at], axis=0))**0.5

        n_larger += np.sum(sim_vals > reference_val)

        estimate = n_larger / n_sims
        if estimate == 0 or estimate == 1:
            continue

        sd = (estimate * (1 - estimate)) ** 0.5
        if np.abs(estimate - compare_val) > 2.576 * sd / n_sims ** 0.5:
            decision = np.sign(estimate - compare_val)
            break
        elif 2.576 * sd / n_sims ** 0.5 < max(min(compare_val, 1 - compare_val) * tol_e, tol_e * 10**-1):
            decision = 0
            break
        else:
            n = int((2.576 * sd / np.abs(estimate - compare_val)) ** 2 - n_sims) + base_step
            n = max(min(n, max_step), base_step)

    return decision

