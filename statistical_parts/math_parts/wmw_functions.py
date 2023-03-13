import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal, norm
from scipy.optimize import root_scalar

from statistical_parts.math_parts.error_spending_simulation import get_sim_nr
from statistical_parts.math_parts.cython_wmw_functions import fixed_MW_CDF, simulate_U_stats, vect_MW_CDF
from statistical_parts.math_parts.wmw_exact_power import check_TypeII, HA_CDF_approximation
from statistical_parts.math_parts.t_test_functions import give_fixed_sample_size as t_sample

# TO DO: improve documentation
"""
example 

sample_sizes = np.array([(3, 4, 5), (3, 4, 5)])
alphas = np.array([(0, 0.1, 0.15), (0.05, 0.1, 0.15), (0.05, 0.1, 0.15), (0.2, 0.3, 0.4), (0, 0, 0.05),
        (0.01, 0.02, 0.03)]).transpose()
betas = np.array([(0.1, 0.2, 0.3), (0, 0.2, 0.3), (0.1, 0.2, 0.3), (0.1, 0.2, 0.3), (0, 0, 0.2), 
        (0.15, 0.18, 0.2)]).transpose()
test_parameters = {'cohens_d': 2}
mode = 'marginally exact'
n_analyses = sample_sizes.shape[1]
n_models = alphas.shape[1]

col_names=['model id'] + ['upper bounds 'f'{i}' for i in range(n_analyses)] + ['lower bounds 'f'{i}' for i in 
    range(n_analyses)] + ['cost h0'] + ['cost ha'] + ['power 'f'{i}' for i in range(n_analyses)] + \
    ['True negatives 'f'{i}' for i in range(n_analyses)]
model_ids = [''f'{i}' for i in range(n_models)]
costs = np.sum(sample_sizes, axis=0)

rel_tol = 10**-3
CI=0.95
force_mode=None

df1, _, _, _ = get_statistics(alphas, betas, sample_sizes, rel_tol, CI, col_names, model_ids, costs, test_parameters, 
    force_mode="marginally exact")
df2, _, _, _ = get_statistics(alphas, betas, sample_sizes, rel_tol, CI, col_names, model_ids, costs, test_parameters, 
    force_mode="simulation")
df3, _, _, _ = get_statistics(alphas, betas, sample_sizes, rel_tol, CI, col_names, model_ids, costs, test_parameters, 
    force_mode="normal power")
df4, _, _, _ = get_statistics(alphas, betas, sample_sizes, rel_tol, CI, col_names, model_ids, costs, test_parameters, 
    force_mode="normal")
"""

# Trigger different method, based on accuracy/ time trade-off
_sample_size_cat = [29, 60, 200]


def get_statistics(alphas, betas, sample_sizes, rel_tol, CI, col_names, model_ids, costs, test_parameters,
                   force_mode=None):
    """
    Args:
        alphas: 2D numpy array with total allowed alpha spending per analysis per model
        betas: 2D numpy array with total allowed beta spending per analysis per model
        sample_sizes: 2D numpy array with sample size per group per analysis
        rel_tol: for simulation: relative tolerance
        CI: for simulation: percentage of confidence interval
        col_names: column names for the output dataframe
        model_ids: model names or id numbers
        costs: 1D numpy array accumulated cost per analysis
        test_parameters: dict containing 'cohens_d', real >0, location shift parameter
        force_mode: for testing purposes, override the default solution mode based on sample size

    Returns:

    """
    total_sample_size = np.sum(sample_sizes[:, -1])
    n_analyses = int(sample_sizes.shape[1])
    n_models = alphas.shape[0]
    sample_sizes = np.transpose(sample_sizes)
    cohens_d = test_parameters['cohens_d']
    costs = costs.reshape(-1)

    if force_mode is not None:
        mode = force_mode
    else:
        mode = None

    if mode == 'marginally exact' or (total_sample_size <= _sample_size_cat[0] and mode is None):
        mode = 'marginally exact'
        message1 = 'Calculations finished: '
        message2 = 'power based on marginally exact asymptotic approximation'
    elif mode == 'simulation' or (total_sample_size <= _sample_size_cat[1] and mode is None):
        if rel_tol <= 10**-4:
            rel_tol = 10**-4
        n_simulations = get_sim_nr(alphas, betas, rel_tol) + 100

        mode = 'simulation'
        message1 = 'Simulations and calculations finished: '
        message2 = 'Power per model based on {} simulations'.format(n_simulations)
    elif mode == 'normal power' or (total_sample_size <= _sample_size_cat[2] and mode is None):
        mode = 'normal power'
        message1 = 'Calculations finished: '
        message2 = 'power based on normal asymptotic approximation'
    elif mode == 'normal' or (total_sample_size > _sample_size_cat[2] and mode is None):
        mode = 'normal'
        message1 = 'Calculations finished: '
        message2 = 'statistics based on normal asymptotic approximation'
    else:
        raise ValueError('mode does not exist')

    # region Do the calculations/ simulations
    # region Initialize basics
    prod_ss = np.prod(sample_sizes, axis=1)

    upper_bounds = np.tile(prod_ss + 1, (n_models, 1))
    lower_bounds = - np.ones((n_models, n_analyses), dtype=int)
    power = np.zeros((n_models, n_analyses))
    true_negatives = np.zeros((n_models, n_analyses))
    costs_h0 = np.ones(n_models) * costs[0]
    costs_ha = np.ones(n_models) * costs[0]
    extra_costs_h0 = np.zeros(n_models)
    extra_costs_ha = np.zeros(n_models)

    h0_normal_upper = np.inf * np.ones((n_models, n_analyses))
    h0_normal_lower = -np.inf * np.ones((n_models, n_analyses))
    ha_normal_upper = np.inf * np.ones((n_models, n_analyses))
    ha_normal_lower = -np.inf * np.ones((n_models, n_analyses))

    p0, p1, p2 = determine_p0_2(cohens_d, pdf='Min ARE')

    if mode == 'normal':
        updated_alphas = alphas.copy()
        updated_betas = betas.copy()
    elif mode == 'normal power':
        updated_alphas = np.zeros((n_models, n_analyses))
        updated_betas = betas
    else:
        updated_alphas = np.zeros((n_models, n_analyses))
        updated_betas = np.zeros((n_models, n_analyses))

    means_h0 = prod_ss / 2
    means_ha = prod_ss * p0

    cov_matrix_h0 = np.zeros((n_analyses, n_analyses))
    cov_matrix_ha = np.zeros((n_analyses, n_analyses))

    for i in range(n_analyses):
        for j in range(i + 1):
            cov_matrix_h0[i, j] = cov_matrix_h0[j, i] = prod_ss[j] * (np.sum(sample_sizes[i, :]) + 1) / 12
            if mode == 'marginally exact':
                cov_matrix_ha[i, j] = cov_matrix_ha[j, i] = \
                    ((sample_sizes[i, 0] - 1) * p1 + (sample_sizes[i, 1] - 1) * p2
                     + p0 + p0 ** 2 * (1 - np.sum(sample_sizes[i, :]))) * prod_ss[j]

    if mode == 'normal power' or mode == 'normal':
        cov_matrix_ha = cov_matrix_h0
    elif mode == 'simulation':
        sims = simulate_U_stats(n_simulations, sample_sizes, cohens_d)

    # function to calculate the probability of a multidimensional interval for the multivariate normal distribution
    def normal_probability(smaller_than_vals, larger_than_vals, null_hypothesis=True):
        for i_1 in range(len(smaller_than_vals)):
            if larger_than_vals[i_1] != -np.inf:
                # p(a cap b) = p(b) - p(not a cap b)

                new_l_vals = larger_than_vals.copy()
                new_l_vals[i_1] = -np.inf
                new_s_vals = smaller_than_vals.copy()
                new_s_vals[i_1] = larger_than_vals[i_1]

                p_b = normal_probability(smaller_than_vals, new_l_vals, null_hypothesis)
                p_not_a_cap_b = normal_probability(new_s_vals, new_l_vals, null_hypothesis)

                return p_b - p_not_a_cap_b
        if null_hypothesis:
            return multivariate_normal.cdf(smaller_than_vals, mean=means_h0[:len(smaller_than_vals)],
                                           cov=cov_matrix_h0[:len(smaller_than_vals), :len(smaller_than_vals)])
        else:
            return multivariate_normal.cdf(smaller_than_vals, mean=means_ha[:len(smaller_than_vals)],
                                           cov=cov_matrix_ha[:len(smaller_than_vals), :len(smaller_than_vals)])

    # make a specific instance of the above function where only the last smaller than/larger than variable can change,
    # so that during root finding we don't have to go through the recursion every time
    def get_normal_probability_func(fixed_s, fixed_l, smaller_than=True, null_hypothesis=True):
        if len(fixed_s) == 0:
            if smaller_than:
                if null_hypothesis:
                    def func(u): return norm.cdf(u, mean=means_h0[0], sd=cov_matrix_h0[0, 0]**0.5)
                else:
                    def func(u): return norm.cdf(u, mean=means_ha[0], sd=cov_matrix_ha[0, 0]**0.5)
            else:
                if null_hypothesis:
                    def func(u): return 1 - norm.cdf(u, mean=means_h0[0], sd=cov_matrix_h0[0, 0]**0.5)
                else:
                    def func(u): return 1 - norm.cdf(u, mean=means_ha[0], sd=cov_matrix_ha[0, 0]**0.5)
            return func

        if not smaller_than:
            a = normal_probability(fixed_s, fixed_l, null_hypothesis)
            b = get_normal_probability_func(fixed_s, fixed_l, True, null_hypothesis)

            def func(u): return a - b(u)
            return func

        for i_1 in range(len(fixed_l)):
            if fixed_l[i_1] != -np.inf:
                # p(a cap b) = p(b) - p(not a cap b)

                new_l_vals = fixed_l.copy()
                new_l_vals[i_1] = -np.inf
                new_s_vals = fixed_s.copy()
                new_s_vals[i_1] = fixed_l[i_1]

                p_b = get_normal_probability_func(fixed_s, new_l_vals, True, null_hypothesis)
                p_not_a_cap_b = get_normal_probability_func(new_s_vals, new_l_vals, True, null_hypothesis)
                def func(u): return p_b(u) - p_not_a_cap_b(u)
                return func

        if null_hypothesis:
            def func(u): return multivariate_normal.cdf(np.append(fixed_s, u), mean=means_h0[:len(fixed_s) + 1],
                                                        cov=cov_matrix_h0[:len(fixed_s) + 1, :len(fixed_s) + 1])
            return func
        else:
            def func(u): return multivariate_normal.cdf(np.append(fixed_s, u), mean=means_ha[:len(fixed_s) + 1],
                                                        cov=cov_matrix_ha[:len(fixed_s) + 1, :len(fixed_s) + 1])
            return func

    # endregion
    # region calculate 1st analysis bounds
    here = alphas[:, 0] > 10 ** -10
    if np.any(here):
        h0_normal_upper[here, 0] = norm.ppf(1 - alphas[here, 0]) * cov_matrix_h0[0, 0] ** 0.5 + means_h0[0]

        if mode != 'normal':
            upper_bounds[here, 0], updated_alphas[here, 0] = \
                transform_h0(sample_sizes[0, 0], sample_sizes[0, 1], alphas[here, 0],
                             np.ceil(h0_normal_upper[here, 0]).astype(int))
            h0_normal_upper[here, 0] = norm.ppf(1 - updated_alphas[here, 0]) * cov_matrix_h0[0, 0] ** 0.5 + means_h0[0]
        else:
            upper_bounds[here, 0] = np.ceil(h0_normal_upper[here, 0]).astype(int)

    here2 = betas[:, 0] > 10 ** -10
    if mode == 'marginally exact':
        if np.any(here2):
            ha_normal_lower[here2, 0] = norm.ppf(betas[here2, 0]) * cov_matrix_ha[0, 0] ** 0.5 + means_ha[0]

            lower_bounds[here2, 0], updated_betas[here2, 0] = \
                transform_ha(n1=sample_sizes[0, 0], n2=sample_sizes[0, 1], marginal_betas=betas[here2, 0],
                             normal_guesses=np.floor(ha_normal_lower[here2, 0]).astype(int), cohens_d=cohens_d)
            ha_normal_lower[here2, 0] = norm.ppf(updated_betas[here2, 0]) * cov_matrix_ha[0, 0] ** 0.5 + means_ha[0]

        if np.any(here):
            c_p = HA_CDF_approximation(np.array(upper_bounds[here, 0]).reshape(-1, 1) - 1, sample_sizes[0, 0].reshape(1),
                                       sample_sizes[0, 1].reshape(1), cohens_d, "Min ARE", max_rows=30)
            ha_normal_upper[here, 0] = norm.ppf(c_p) * cov_matrix_ha[0, 0] ** 0.5 + means_ha[0]
    elif mode == 'simulation':
        undecided = np.ones((n_models, n_simulations), dtype='?')

        for i in range(n_models):
            if betas[i, 0] > 10 ** -10:
                lower_bounds[i, 0] = np.round(np.quantile(np.array(sims[0, :]), betas[i, 0])) - 1
                if lower_bounds[i, 0] + 1 >= upper_bounds[i, 0]:
                    updated_betas[i, :] = np.sum(upper_bounds[i, 0] - 1 >= sims[0, :]) / n_simulations
                else:
                    updated_betas[i, 0] = np.sum(lower_bounds[i, 0] >= sims[0, :]) / n_simulations
            power[i, :] = np.sum(sims[0, :] >= upper_bounds[i, 0]) / n_simulations
            undecided[i, :] = np.logical_and(lower_bounds[i, 0] < sims[0, :], sims[0, :] < upper_bounds[i, 0])
    elif mode == 'normal power':
        ha_normal_lower[here2, 0] = norm.ppf(betas[here2, 0]) * cov_matrix_ha[0, 0] ** 0.5 + means_ha[0]
        lower_bounds[here2, 0] = np.floor(ha_normal_lower[here2, 0]).astype(int)

        if np.any(here):
            ha_normal_upper[here, 0] = h0_normal_upper[here, 0]
    else:
        ha_normal_lower[here2, 0] = norm.ppf(betas[here2, 0]) * cov_matrix_ha[0, 0] ** 0.5 + means_ha[0]
        h0_normal_lower[here2, 0] = ha_normal_lower[here2, 0]
        lower_bounds[here2, 0] = np.floor(ha_normal_lower[here2, 0]).astype(int)

        if np.any(here):
            ha_normal_upper[here, 0] = h0_normal_upper[here, 0]

    if mode != 'normal' and np.any(here2):
        c_p = vect_MW_CDF(lower_bounds[here2, 0], sample_sizes[0, 0], sample_sizes[0, 1])
        h0_normal_lower[here2, 0] = norm.ppf(c_p) * cov_matrix_h0[0, 0] ** 0.5 + means_h0[0]
    # endregion
    # region evaluate models that are done
    done = lower_bounds[:, 0] + 1 >= upper_bounds[:, 0]
    if np.any(done) or n_analyses == 1:
        updated_alphas[done, :] = updated_alphas[done, 0].reshape(-1, 1)

        if mode == 'marginally exact':
            c_p = HA_CDF_approximation(np.array(lower_bounds[done, 0]).reshape(-1, 1), sample_sizes[0, 0].reshape(1),
                                       sample_sizes[0, 1].reshape(1), cohens_d, "Min ARE", max_rows=30)
            updated_betas[done, :] = np.tile(np.array(c_p).reshape((-1, 1)), (1, n_analyses))
        elif mode == 'simulation':
            pass
        else:
            updated_betas[done, :] = np.tile(
                norm.cdf((lower_bounds[done, 0] - means_ha[0])/cov_matrix_ha[0, 0] ** 0.5).reshape((-1, 1)),
                (1, n_analyses))

        true_negatives[done, :] = 1 - updated_alphas[done, :]
        power[done, :] = 1 - updated_betas[done, :]
    # endregion
    not_done = np.logical_not(done)

    true_negatives[not_done, :] = norm.cdf((h0_normal_lower[not_done, 0] - means_h0[0]) /
                                           cov_matrix_h0[0, 0] ** 0.5)[:, np.newaxis]
    if mode != 'simulation':
        power[not_done, :] = 1 - norm.cdf((ha_normal_upper[not_done, 0] - means_ha[0]) /
                                          cov_matrix_ha[0, 0] ** 0.5)[:, np.newaxis]

    for i in np.arange(1, n_analyses):
        # extra cost = added cost of performing ith analysis * probability of getting to the ith analysis
        extra_costs_h0[:] = extra_costs_ha[:] = 0
        for j in np.arange(n_models)[not_done]:
            extra_costs_h0[j] = (costs[i] - costs[i-1]) * normal_probability(h0_normal_upper[j, :i],
                                                                             h0_normal_lower[j, :i], True)
            if mode != 'simulation':
                extra_costs_ha[j] = (costs[i] - costs[i-1]) * normal_probability(ha_normal_upper[j, :i],
                                                                                 ha_normal_lower[j, :i], False)
        here = np.arange(n_models)[not_done][alphas[not_done, i] - updated_alphas[not_done, i - 1] > 10 ** -10]
        if here.size > 0:
            funcs = {f'{j}': get_normal_probability_func(fixed_s=h0_normal_upper[j, :i], fixed_l=h0_normal_lower[j, :i],
                                                         smaller_than=False, null_hypothesis=True) for j in here}
            for j in here:
                def find_this_h0(u):
                    return alphas[j, i] - updated_alphas[j, i - 1] - funcs[f'{j}'](u)

                guess1 = norm.ppf(1 - alphas[j, i] + updated_alphas[j, i - 1]) * cov_matrix_h0[i, i] ** 0.5 + \
                    means_h0[i]
                h0_normal_upper[j, i] = find_root(guess1, find_this_h0)
                if h0_normal_upper[j, i] == -np.inf:
                    # in the last analysis, everything is significant
                    # -> useless, just make it significant in previous analysis
                    extra_costs_h0[j] = 0
                    extra_costs_ha[j] = 0

                    here = here[here != j]
                    not_done[j] = False
                    upper_bounds[j, i - 1] = lower_bounds[j, i - 1] + 1
                    if mode != 'normal':
                        c_p = fixed_MW_CDF(upper_bounds[j, i - 1] - 1, sample_sizes[i - 1, 0], sample_sizes[i - 1, 1])
                        h0_normal_upper[j, i - 1] = norm.ppf(c_p) * cov_matrix_h0[i - 1, i - 1] ** 0.5 + means_h0[i - 1]
                    else:
                        h0_normal_upper[j, i - 1] = h0_normal_lower[j, i - 1]
                    lo = h0_normal_lower[j, :i].copy()
                    lo[i - 1] = h0_normal_upper[j, i - 1]
                    up = h0_normal_upper[j, :i].copy()
                    up[i - 1] = np.inf

                    if i == 1:
                        updated_alphas[j, :] = normal_probability(up, lo, True)
                    else:
                        updated_alphas[j, (i - 1):] = updated_alphas[j, i - 2] + normal_probability(up, lo, True)

            if mode != 'normal':
                c_p = norm.cdf((h0_normal_upper[here, i] - means_h0[i]) / cov_matrix_h0[i, i] ** 0.5)
                upper_bounds[here, i], qs = transform_h0(sample_sizes[i, 0], sample_sizes[i, 1], c_p,
                                                         np.ceil(h0_normal_upper[here, i]).astype(int))

                h0_normal_upper[here, i] = norm.ppf(1 - qs) * cov_matrix_h0[i, i] ** 0.5 + means_h0[i]
                for j in here:
                    updated_alphas[j, i] = updated_alphas[j, i - 1] + funcs[f'{j}'](h0_normal_upper[j, i])

            else:
                upper_bounds[here, i] = np.ceil(h0_normal_upper[here, i]).astype(int)
                upper_bounds[upper_bounds < 0] = -1

        here2 = np.arange(n_models)[not_done][betas[not_done, i] - updated_betas[not_done, i - 1] > 10 ** -10]
        if here2.size > 0 and mode != 'simulation':
            funcs = {f'{j}': get_normal_probability_func(ha_normal_upper[j, :i], ha_normal_lower[j, :i],
                                                         smaller_than=True, null_hypothesis=False) for j in here2}
            for j in here2:
                def find_this_ha(u):
                    return betas[j, i] - updated_betas[j, i - 1] - funcs[f'{j}'](u)

                guess1 = norm.ppf(betas[j, i] - updated_betas[j, i - 1]) * cov_matrix_ha[i, i] ** 0.5 + means_ha[i]
                ha_normal_lower[j, i] = find_root(guess1, find_this_ha, f_increasing=False)

                if ha_normal_lower[j, i] == np.inf:
                    # done
                    lower_bounds[j, i] = upper_bounds[j, i] - 1
                    here2 = here2[here2 != j]
                    ha_normal_lower[j, i] = ha_normal_upper[j, i]

            if mode == 'marginally exact':
                c_p = norm.cdf((ha_normal_lower[here2, i] - means_ha[i]) / cov_matrix_ha[i, i] ** 0.5)
                lower_bounds[here2, i], qs = \
                    transform_ha(n1=sample_sizes[i, 0], n2=sample_sizes[i, 1], marginal_betas=c_p,
                                 normal_guesses=np.floor(ha_normal_lower[here2, i]).astype(int), cohens_d=cohens_d)

                ha_normal_lower[here2, i] = norm.ppf(qs) * cov_matrix_ha[i, i] ** 0.5 + means_ha[i]
                for j in here2:
                    updated_betas[j, i] = updated_betas[j, i - 1] + funcs[f'{j}'](ha_normal_lower[j, i])

                if here.size > 0:
                    c_p = HA_CDF_approximation(np.array(upper_bounds[here, i]).reshape(-1, 1) - 1,
                                               sample_sizes[i, 0].reshape(1),
                                               sample_sizes[i, 1].reshape(1), cohens_d, "Min ARE", max_rows=30)
                    ha_normal_upper[here, i] = norm.ppf(c_p) * cov_matrix_ha[i, i] ** 0.5 + means_ha[i]

            else:
                h0_normal_lower[here2, i] = ha_normal_lower[here2, i]
                lower_bounds[here2, i] = np.floor(ha_normal_lower[here2, i]).astype(int)
                lower_bounds[lower_bounds > np.prod(sample_sizes[i, :])] = np.prod(sample_sizes[i, :])
                if here.size > 0:
                    ha_normal_upper[here, i] = h0_normal_upper[here, i]
        elif mode == 'simulation':
            for j in np.arange(n_models)[not_done]:
                p_left = np.sum(undecided[j, :])/n_simulations
                if 10 ** -10 < betas[j, i] - updated_betas[j, i - 1] < p_left:
                    lower_bounds[j, i] = np.round(np.quantile(np.array(sims[i, undecided[j, :]]),
                                                              (betas[j, i] - updated_betas[j, i - 1]) * p_left)) - 1
                    if lower_bounds[j, i] + 1 >= upper_bounds[j, i] or i == n_analyses-1:
                        updated_betas[j, i:] = np.sum(upper_bounds[j, i] - 1 >= sims[i, undecided[j, :]]) \
                                               / n_simulations + updated_betas[j, i - 1]
                    else:
                        updated_betas[j, i] = np.sum(lower_bounds[j, i] >= sims[i, undecided[j, :]]) / n_simulations \
                                              + updated_betas[j, i - 1]

                elif p_left <= betas[j, i] - updated_betas[j, i - 1]:
                    lower_bounds[j, i] = upper_bounds[j, i] - 1

                costs_ha[j] += (costs[i] - costs[i-1]) * p_left
                power[j, i:] += np.sum(sims[i, undecided[j, :]] >= upper_bounds[j, i]) / n_simulations

                undecided[j, undecided[j, :]] = np.logical_and(lower_bounds[j, i] < sims[i, undecided[j, :]],
                                                               sims[i, undecided[j, :]] < upper_bounds[j, i])

        costs_h0 += extra_costs_h0
        costs_ha += extra_costs_ha

        if mode != 'normal' and here2.size > 0:
            c_p = vect_MW_CDF(lower_bounds[here2, i], sample_sizes[i, 0], sample_sizes[i, 1])
            h0_normal_lower[here2, i] = norm.ppf(c_p) * cov_matrix_h0[i, i] ** 0.5 + means_h0[i]

        for j in np.arange(n_models)[not_done]:
            lo = h0_normal_lower[j, :].copy()
            lo[i] = -np.inf
            up = h0_normal_upper[j, :].copy()
            up[i] = h0_normal_lower[j, i]
            true_negatives[j, i:] = true_negatives[j, i - 1] + normal_probability(up, lo, True)
            if mode != 'simulation':
                lo = ha_normal_lower[j, :].copy()
                lo[i] = ha_normal_upper[j, i]
                up = ha_normal_upper[j, :].copy()
                up[i] = np.inf
                power[j, i] = power[j, i - 1] + normal_probability(up, lo, False)

        # region evaluate models that are done
        if i == n_analyses - 1:
            new_done = not_done
        else:
            new_done = np.logical_and(lower_bounds[:, i] + 1 >= upper_bounds[:, i], not_done)
        lower_bounds[new_done, i] = upper_bounds[new_done, i] - 1

        if np.any(new_done):
            c_p = vect_MW_CDF(lower_bounds[new_done, i], sample_sizes[i, 0], sample_sizes[i, 1])
            h0_normal_lower[new_done, i] = norm.ppf(c_p) * cov_matrix_h0[i, i] ** 0.5 + means_h0[i]

            for j in np.arange(n_models)[new_done]:
                lo = h0_normal_lower[j, :].copy()
                lo[i] = -np.inf
                up = h0_normal_upper[j, :].copy()
                up[i] = h0_normal_lower[j, i]
                true_negatives[j, i:] = true_negatives[j, i - 1] + normal_probability(up, lo, True)
                if mode != 'simulation':
                    lo = ha_normal_lower[j, :].copy()
                    lo[i] = ha_normal_upper[j, i]
                    up = ha_normal_upper[j, :].copy()
                    up[i] = np.inf
                    power[j, i] = power[j, i - 1] + normal_probability(up, lo, False)

            if mode == 'marginally exact':
                c_p = HA_CDF_approximation(np.array(lower_bounds[new_done, i]).reshape(-1, 1),
                                           sample_sizes[i, 0].reshape(1),
                                           sample_sizes[i, 1].reshape(1), cohens_d, "Min ARE", max_rows=30)
                ha_normal_lower[new_done, i] = norm.ppf(c_p) * cov_matrix_ha[i, i] ** 0.5 + means_ha[i]

                for j in np.arange(n_models)[new_done]:
                    lo = ha_normal_lower[j, :].copy()
                    lo[i] = -np.inf
                    up = ha_normal_upper[j, :].copy()
                    up[i] = ha_normal_lower[j, i]
                    updated_betas[j, i:] = updated_betas[j, i - 1] + normal_probability(up, lo, False)

                    lo[i] = ha_normal_lower[j, i]
                    up[i] = np.inf
                    power[j, i:] = power[j, i - 1] + normal_probability(up, lo, False)

            elif mode == 'simulation':
                pass
            else:
                for j in np.arange(n_models)[new_done]:
                    lo = ha_normal_lower[j, :].copy()
                    lo[i] = -np.inf
                    up = ha_normal_upper[j, :].copy()
                    up[i] = lower_bounds[j, i]
                    updated_betas[j, i] = updated_betas[j, i - 1] + normal_probability(up, lo, False)

                    lo[i] = lower_bounds[j, i]
                    up[i] = np.inf
                    power[j, i:] = power[j, i - 1] + normal_probability(up, lo, False)
            not_done[not_done] = lower_bounds[not_done, i] + 1 <= upper_bounds[not_done, i]
        # endregion
    # endregion

    # Users won't realise that upper bounds = prod_ss + 1 and lower bounds = -1 are useless -> set to nan
    upper_bounds = upper_bounds.astype(float)
    lower_bounds = lower_bounds.astype(float)
    upper_bounds[upper_bounds == prod_ss[np.newaxis, :] + 1] = np.nan
    lower_bounds[lower_bounds == -1] = np.nan

    estimates = pd.DataFrame(np.concatenate((
        np.arange(0, n_models).reshape(-1, 1), upper_bounds, lower_bounds, costs_h0.reshape(-1, 1),
        costs_ha.reshape(-1, 1), power, true_negatives), axis=1), columns=col_names)
    std_errors = pd.DataFrame(np.concatenate((
        np.arange(0, n_models).reshape(-1, 1), np.nan + np.zeros((n_models, n_analyses * 4 + 2))), axis=1),
        columns=col_names)
    estimates['Model id'] = model_ids
    std_errors['Model id'] = model_ids

    return estimates, std_errors, message1, message2


def give_fixed_sample_size(cohens_d, alpha, beta, sides):
    if sides == 'one':
        s_f = 1
    else:
        s_f = 2

    alpha = alpha*0.99999999     # significance for p-value <= 0.05 or < 0.05, this way no discussion

    def eval_typeII(n):
        # normal approximation of the critical value
        crit_guess = int(norm.ppf(1 - alpha / s_f) * (n**2 * (2 * n + 1) / 12)**0.5 + 0.5 * n**2)
        if 2*n < _sample_size_cat[1]:
            # get the exact critical value and simulate power
            crit_val, _ = transform_h0(n, n, np.array([alpha / s_f]), np.array([crit_guess]))
            n_sims = int(min(100000, 1 / beta * 100) + 1000)
            sims = simulate_U_stats(n_sims, sample_sizes=np.array([(n, n)]), cohens_d=cohens_d, pdf='Min ARE')
            tII = np.sum(sims < crit_val) / n_sims

        elif 2*n < _sample_size_cat[2]:
            # get the exact critical value and asymptotic power
            p0, p1, p2 = determine_p0_2(cohens_d, pdf='Min ARE')
            crit_val = transform_h0(n, n, np.array([alpha/s_f]), np.array([crit_guess]))[0][0]
            tII = norm.cdf((crit_val - p0 * n ** 2) / (n**2*(2*n+1)/12)**0.5)

        else:
            # asymptotic everything
            p0, p1, p2 = determine_p0_2(cohens_d, pdf='Min ARE')
            crit_val = norm.ppf(1 - alpha / s_f) * (n**2 * (2 * n + 1) / 12)**0.5 + 0.5 * n**2 \
                + 0.5 * (alpha/s_f > 0.02)  # continuity correction if necessary
            tII = norm.cdf((crit_val - p0 * n ** 2) / (n**2*(2*n+1)/12)**0.5)

        return tII

    n_guess = int(t_sample(cohens_d, alpha, beta, sides)[0])
    typeII = eval_typeII(n_guess)
    while typeII > beta:
        n_guess = n_guess + 1
        typeII = eval_typeII(n_guess)

    return n_guess, typeII


def get_p_equivalent(x, N, sig):
    if np.isnan(x):
        return np.nan

    n1 = N[0]
    n2 = N[1]
    if n1 + n2 < _sample_size_cat[2]:
        actual = 1 - fixed_MW_CDF(x - 1, n1, n2)

        # This bit is to make it look nicer
        # It writes the cut-off points with the minimum nr of decimals required
        # It also avoids confusion of <= or < p for significance (resp. >= or > for futility)
        if sig:
            nxt = 1 - fixed_MW_CDF(x - 2, n1, n2)
        else:
            nxt = 1 - fixed_MW_CDF(x, n1, n2)
        mid = (actual + nxt) / 2
        if actual == nxt:
            return actual
        nice = np.round(mid, -np.floor(np.log10(np.abs(actual - nxt))).astype(int))
        return nice
    else:
        return 1 - norm.cdf((x - 0.5 * n1 * n2)/(n1 * n2 * (n1 + n2 + 1) / 12)**0.5)


# region help functions
def determine_p0_2(cohens_d, pdf='Min ARE'):
    # function to determine
    # p0; the probability that one sample from the first group is larger than one from the second
    # p1; the probability that one sample from the first group is larger than two from the second
    # p2; the probability that two samples from the first group are larger than one from the second

    # overlapping domain of X and Y = A
    if cohens_d >= 2 * 5 ** 0.5:
        p0 = p1 = p2 = 1
    else:
        b = 3 / 20 / (5 ** 0.5)
        xNinA = 1/100 * cohens_d**2 * (15 - cohens_d * 5**0.5)  # = yNinA
        xLargerYInA = (19 * cohens_d**6 - 120 * 5**0.5 * cohens_d**5 + 1050 * cohens_d**4 + 400 * 5**0.5 *
                       cohens_d**3 - 12000 * cohens_d**2 + 4800 * 5**0.5 * cohens_d + 20000)/40000
        xLargerYyInA = ((56000 * 5**0.5 + 156800 * cohens_d + 9840 * 5**0.5 * cohens_d**2 + 2000 * cohens_d**3 +
                         3688 * 5**0.5 * cohens_d**4 + 1068 * cohens_d**5 - 416 * 5**0.5 * cohens_d**6 +
                         85 * cohens_d**7)/4536 + ((3 * 5**0.5 - cohens_d) * cohens_d**2 *
                                                   (-1000 - 440 * 5**0.5 * cohens_d - 90 * cohens_d**2 +
                                                                  4 * 5**0.5 * cohens_d**3 + cohens_d**4))/540) * \
            b**3 * (2 * 5**0.5 - cohens_d)**2
        xXLargerYInA = ((-5 * (-2000 * 5**0.5 - 6000 * cohens_d + 600 * 5**0.5 * cohens_d**2 + 2600 * cohens_d**3
                               - 285 * 5**0.5 * cohens_d**4 - 255 * cohens_d**5 + 45 * 5**0.5 * cohens_d**6
                               - 9 * cohens_d**7))/54 +
                        ((10000 * 5**0.5)/9 - 2000 * cohens_d - (17400 * 5**0.5 * cohens_d**2)/7 + 2600 *
                         cohens_d**3 + 585 * 5**0.5 * cohens_d**4 - 825 * cohens_d**5 +
                         (74 * 5**0.5 * cohens_d**6)/3 + 34 * cohens_d**7 - (29 * cohens_d**8)/(2 * 5**0.5) +
                         (383 * cohens_d**9)/1260)/18) * b**3

        p0 = xNinA + (1-xNinA) * xNinA + xLargerYInA
        p1 = xNinA + xNinA ** 2 * (1 - xNinA) + 2 * xNinA * xLargerYInA + 2 * xLargerYyInA
        p2 = xNinA + xNinA ** 2 * (1 - xNinA) + 2 * xNinA * xLargerYInA + 2 * xXLargerYInA
    return p0, p1, p2


def transform_h0(n1, n2, marginal_alphas, normal_guesses):
    normal_guesses[normal_guesses < 0] = -1
    normal_guesses[normal_guesses > n1*n2] = n1*n2
    marginal_alphas = marginal_alphas
    # p_1 contains the probabilities of the u statistics, p_2 will contain the probabilities of one value lower
    u_1 = normal_guesses.copy()
    p_1 = 1 - vect_MW_CDF(normal_guesses - 1, n1, n2).astype(float)
    p_2 = p_1.copy()

    u_1[p_1 > marginal_alphas + 10 ** -10] += 1
    p_1[p_1 > marginal_alphas + 10 ** -10] = 1 - vect_MW_CDF(u_1[p_1 > marginal_alphas + 10 ** -10] - 1, n1, n2)
    p_2[p_1 <= marginal_alphas + 10 ** -10] = 1 - vect_MW_CDF(u_1[p_1 <= marginal_alphas + 10 ** -10] - 2, n1, n2)

    done = np.logical_or(np.abs(p_1 - marginal_alphas) <= 10 ** -10,
                         np.logical_and(p_1 < marginal_alphas - 10 ** -10,
                                        p_2 > marginal_alphas + 10 ** -10))
    while not np.all(done):
        too_high = p_1 > marginal_alphas + 10 ** -10
        u_1[too_high] += 1
        p_2[too_high] = p_1[too_high]
        p_1[too_high] = 1 - vect_MW_CDF(u_1[too_high] - 1, n1, n2)

        too_low = p_2 < marginal_alphas - 10 ** -10
        u_1[too_low] -= 1
        p_1[too_low] = p_2[too_low]
        p_2[too_low] = 1 - vect_MW_CDF(u_1[too_low] - 2, n1, n2)

        good = np.abs(p_2 - marginal_alphas) <= 10 ** -10
        u_1[good] -= 1
        p_1[good] = p_2[good]
        p_2[good] = np.inf

        done = np.logical_or(np.abs(p_1 - marginal_alphas) <= 10 ** -10,
                             np.logical_and(p_1 < marginal_alphas - 10 ** -10,
                                            p_2 > marginal_alphas + 10 ** -10))
    return u_1, p_1


def transform_ha(n1, n2, marginal_betas, normal_guesses, cohens_d, tol=10**-5):
    # it is more efficient to check a few values at once, because of shared caching within the iteration
    n_guesses = 3
    guesses = min(max(np.median(normal_guesses), 1), n1*n2)
    guesses = np.ones((n_guesses, 1), dtype=int) * np.median(guesses).astype(int)
    guesses[0] -= 1
    guesses[2] += 1

    n_vals = marginal_betas.size
    result_crit = np.ones(n_vals, dtype=int)
    result_ps = np.ones(n_vals)

    results = np.array(check_TypeII(guesses, np.array(n1).reshape(1), np.array(n2).reshape(1), cohens_d,
                                    "Min ARE", marginal_betas.reshape(-1), max_rows=30, solution="lower", tol=tol))
    too_high = []
    too_low = []

    for i in range(n_vals):
        if np.any(np.abs(marginal_betas[i] - results) <= tol * marginal_betas[i]):

            result_crit[i] = guesses[np.abs(marginal_betas[i] - results) <= tol * marginal_betas[i]][-1]
            result_ps[i] = results[np.abs(marginal_betas[i] - results) <= tol * marginal_betas[i]][-1]
        elif np.any(results <= marginal_betas[i] + tol * marginal_betas[i]) and \
                np.any(results > marginal_betas[i] - tol * marginal_betas[i]):

            result_crit[i] = guesses[results <= marginal_betas[i] + tol * marginal_betas[i]][-1]
            result_ps[i] = results[results <= marginal_betas[i] + tol * marginal_betas[i]][-1]
        elif np.all(results > marginal_betas[i] + tol * marginal_betas[i]):
            too_high.append(i)
        else:
            too_low.append(i)

    guess_h = guesses.copy()
    while too_high:
        guess_h -= n_guesses
        results = np.array(check_TypeII(guess_h, np.array(n1).reshape(1), np.array(n2).reshape(1), cohens_d,
                                        "Min ARE", marginal_betas[too_high], max_rows=30, solution="lower", tol=tol))
        for i in too_high:
            if np.any(np.abs(marginal_betas[i] - results) <= tol * marginal_betas[i]):
                result_crit[i] = guess_h[np.abs(marginal_betas[i] - results) < tol * marginal_betas[i]][-1]
                result_ps[i] = results[np.abs(marginal_betas[i] - results) < tol * marginal_betas[i]][-1]

                too_high.remove(i)
            elif results[-1] <= marginal_betas[i] + tol * marginal_betas[i]:
                result_crit[i] = guess_h[results <= marginal_betas[i] + tol * marginal_betas[i]][-1]
                result_ps[i] = HA_CDF_approximation(np.array([result_crit[i]]), np.array(n1).reshape(1),
                                                    np.array(n2).reshape(1), cohens_d, "Min ARE", max_rows=30,
                                                    tol=tol)[0]

                too_high.remove(i)
            elif np.any(results <= marginal_betas[i] + tol * marginal_betas[i]):
                result_crit[i] = guess_h[results <= marginal_betas[i] + tol * marginal_betas[i]][-1]
                result_ps[i] = results[results <= marginal_betas[i] + tol * marginal_betas[i]][-1]

                too_high.remove(i)

    while too_low:
        guess_h = guesses[-1].copy()
        guesses += n_guesses
        results = np.array(check_TypeII(guesses, np.array(n1).reshape(1), np.array(n2).reshape(1), cohens_d,
                                        "Min ARE", marginal_betas[too_low], max_rows=30, solution="lower", tol=tol))

        for i in too_low:
            if np.any(np.abs(marginal_betas[i] - results) <= tol * marginal_betas[i]):
                result_crit[i] = guesses[np.abs(marginal_betas[i] - results) < tol * marginal_betas[i]][-1]
                result_ps[i] = results[np.abs(marginal_betas[i] - results) < tol * marginal_betas[i]][-1]

                too_low.remove(i)
            elif results[0] > marginal_betas[i] - tol * marginal_betas[i]:
                result_crit[i] = guess_h
                result_ps[i] = HA_CDF_approximation(guess_h, np.array(n1).reshape(1), np.array(n2).reshape(1),
                                                    cohens_d, "Min ARE", max_rows=30, tol=tol)[0]

                too_low.remove(i)
            elif np.any(results > marginal_betas[i] - tol * marginal_betas[i]):
                result_crit[i] = guesses[results <= marginal_betas[i] + tol * marginal_betas[i]][-1]
                result_ps[i] = results[results <= marginal_betas[i] + tol * marginal_betas[i]][-1]

                too_low.remove(i)

    return result_crit, result_ps


def find_root(guess, func, f_increasing=True, tol=10**-5):
    now = func(guess)
    flag = True
    okay = False
    if np.abs(now) < tol:
        new_u = guess
        flag = False
        okay = True
    elif now < tol:
        new_dif = 1
        for _ in range(20):
            if f_increasing:
                guess2 = 2 * guess + 1
            else:
                guess2 = guess - new_dif
            now = func(guess2)
            if np.abs(now) < 10 ** -5:
                new_u = guess2
                flag = False
                okay = True
                break
            elif now > tol:
                okay = True
                break
            guess = guess2
            new_dif = 2 * (guess - guess2)
    else:
        new_dif = 1
        for _ in range(20):
            if f_increasing:
                guess2 = guess - new_dif
            else:
                guess2 = 2 * guess + 1
            now = func(guess2)
            if np.abs(now) < tol:
                new_u = guess2
                flag = False
                okay = True
                break
            elif now < tol:
                okay = True
                break
            new_dif = 2 * (guess - guess2)
            guess = guess2
    if okay:
        if flag:
            new_u = root_scalar(func, bracket=[guess, guess2]).root
    else:
        if guess2 > 0:
            new_u = np.inf
        else:
            new_u = -np.inf
    return new_u
# endregion

