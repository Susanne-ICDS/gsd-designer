import numpy as np
from scipy.stats import multivariate_normal, norm
from scipy.stats import t
from scipy.stats import nct
from scipy.optimize import root_scalar

from statistical_parts.math_parts.error_spending_simulation import get_sim_nr
from statistical_parts.math_parts.cython_wmw_functions import fixed_MW_CDF, simulate_U_stats, vect_MW_CDF
from statistical_parts.math_parts.wmw_exact_power import check_TypeII, HA_CDF_approximation

# TO DO: improve documentation
"""
example 

sample_sizes = np.array([(3, 3), (4, 4), (5, 5)])
target_alphas = np.array([(0, 0.2, 0.3), (0.1, 0.2, 0.3), (0.1, 0.2, 0.3)])
target_betas = np.array([(0.1, 0.2, 0.3), (0, 0.2, 0.3), (0.1, 0.2, 0.3)])
cohens_d = 2
mode = 'marginally exact'

n1 = n2 = 3
marginal_alphas = np.array([0.1, 0.2, 0.1])
marginal_betas = np.array([0.1, 0.2, 0.1])
normal_guesses = np.array([0, 2, 3])
"""


def get_statistics(alphas, betas, sample_sizes, rel_tol, CI, col_names, model_ids, default_n_repeats,
                   max_n_repeats, costs, test_parameters, memory_limit):
    total_sample_size = np.sum(sample_sizes[:, -1])
    n_analyses = int(sample_sizes.shape[1])

    if total_sample_size <= 30:
        estimates = get_transformed(sample_sizes, alphas, betas, test_parameters['cohens_d'], col_names, model_ids,
                                    costs, mode='marginally exact')

        return estimates, 'na', 'Calculations finished: ', 'power based on marginally exact asymptotic approximation'

    elif total_sample_size <= 60:
        estimates, std_errors, n_simulations, counts = get_transformed(
            sample_sizes, alphas, betas, test_parameters['cohens_d'], col_names, model_ids, costs, mode='simulation',
            default_n_repeats=default_n_repeats, max_n_repeats=max_n_repeats, rel_tol=rel_tol, CI=CI)

        counts_str = '{}'.format(counts[col_names[1]][0])
        for i in range(counts.shape[0] - 1):
            counts_str += ', ' + '{}'.format(counts[col_names[1]][i])

        return estimates, std_errors, 'Simulations and calculations finished: ', \
            'Power per model based on respectively ' + counts_str + \
            ' estimates with {} simulations each'.format(n_simulations)

    elif total_sample_size <= 200:
        estimates, std_errors, n_simulations, counts = get_transformed(
            sample_sizes, alphas, betas, test_parameters['cohens_d'], col_names, model_ids, costs,
            mode='normal power')
        return estimates, 'na', 'Calculations finished: ', 'power based on normal asymptotic approximation'
    else:
        estimates, std_errors, n_simulations, counts = get_transformed(
            sample_sizes, alphas, betas, test_parameters['cohens_d'], col_names, model_ids, costs,
            mode='normal')
        return estimates, 'na', 'Calculations finished: ', 'statistics based on normal asymptotic approximation'


def get_transformed(sample_sizes, target_alphas, target_betas, cohens_d, col_names, model_ids, costs,
                    mode='marginally exact', default_n_repeats=0, max_n_repeats=0, rel_tol=0, CI=0,
                    prev_obtained_upper=None, prev_obtained_lower=None):
    """
    Args:
        sample_sizes: 2D numpy array with sample size per group per analysis
        target_alphas: 2D numpy array with total allowed alpha spending per analysis per model
        target_betas: 2D numpy array with total allowed beta spending per analysis per model
        cohens_d: real >0, location shift parameter
        col_names:
        model_ids:
        costs: 1D numpy array accumulated cost per analysis
        mode: ['marginally exact', 'simulation', 'normal power', 'normal']
        default_n_repeats:
        max_n_repeats:
        rel_tol:
        CI:
        prev_obtained_upper:
        prev_obtained_lower:

    Returns:

    """
    prod_ss = np.prod(sample_sizes, axis=1)
    n_analyses = prod_ss.size
    n_models = target_alphas.shape[0]

    upper_bounds = np.tile(prod_ss + 1, (n_models, 1))
    lower_bounds = - np.ones((n_models, n_analyses), dtype=int)

    h0_normal_upper = np.inf * np.ones((n_models, n_analyses))
    h0_normal_lower = -np.inf * np.ones((n_models, n_analyses))
    ha_normal_upper = np.inf * np.ones((n_models, n_analyses))
    ha_normal_lower = -np.inf * np.ones((n_models, n_analyses))

    p0, p1, p2 = determine_p0_2(cohens_d, pdf='Min ARE')

    updated_alphas = target_alphas.copy()
    updated_betas = target_betas.copy()

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

    if mode == 'asymptotic power' or mode == 'normal':
        cov_matrix_ha = cov_matrix_h0
    elif mode == 'simulation':
        if rel_tol == 0:
            rel_tol = 0.01
        n = get_sim_nr(target_alphas, target_betas, rel_tol)
        sims = simulate_U_stats(n + 100, sample_sizes, cohens_d)

    def normal_probability(smaller_than_vals, larger_than_vals, null_hypothesis=True):
        for i_1 in range(n_analyses):
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
            return multivariate_normal.cdf(smaller_than_vals, mean=means_h0, cov=cov_matrix_h0)
        else:
            return multivariate_normal.cdf(smaller_than_vals, mean=means_ha, cov=cov_matrix_ha)

    # region calculate 1st analysis bounds
    here = target_alphas[:, 0] > 10 ** -10
    if np.any(here):
        h0_normal_upper[here, 0] = norm.ppf(1 - target_alphas[here, 0]) * cov_matrix_h0[0, 0] ** 0.5 + means_h0[0]

        if mode != 'normal':
            mirror, updated_alphas[here, 0] = \
                transform_h0(sample_sizes[0, 0], sample_sizes[0, 1], target_alphas[here, 0],
                             np.ceil(h0_normal_upper[here, 0]).astype(int))
            upper_bounds[here, 0] = np.prod(sample_sizes[0, :]) - mirror
            h0_normal_upper[here, 0] = norm.ppf(1 - updated_alphas[here, 0]) * cov_matrix_h0[0, 0] ** 0.5 + means_h0[0]
        else:
            upper_bounds[here, 0] = np.ceil(h0_normal_upper[here, 0]).astype(int)

    here2 = target_betas[:, 0] > 10 ** -10
    if mode == 'marginally exact':
        if np.any(here2):
            ha_normal_lower[here2, 0] = norm.ppf(target_betas[here2, 0]) * cov_matrix_ha[0, 0] ** 0.5 + means_ha[0]

            lower_bounds[here2, 0], updated_betas[here2, 0] = \
                transform_ha(n1=sample_sizes[0, 0], n2=sample_sizes[0, 1], marginal_betas=target_betas[here2, 0],
                             normal_guesses=np.floor(ha_normal_lower[here2, 0]).astype(int), cohens_d=cohens_d)
            ha_normal_lower[here2, 0] = norm.ppf(updated_betas[here2, 0]) * cov_matrix_ha[0, 0] ** 0.5 + means_ha[0]

        if np.any(here):
            c_p = HA_CDF_approximation(np.array(upper_bounds[here, 0]).reshape(-1, 1) - 1, sample_sizes[0, 0].reshape(1),
                                       sample_sizes[0, 1].reshape(1), cohens_d, "Min ARE", max_rows=30)
            ha_normal_upper[here, 0] = norm.ppf(c_p) * cov_matrix_ha[0, 0] ** 0.5 + means_ha[0]

    elif mode == 'simulation':
        undecided = np.ones((n_models, n), dtype='?')

        for i in range(n_models):
            if target_betas[i, 0] > 10 ** -10:
                lower_bounds[i, 0] = np.round(np.quantile(np.array(sims[0, :]), target_betas[i, 0])) - 1
                updated_betas[i, 0] = np.sum(lower_bounds[i, 0] >= sims[0, :]) / n

            undecided[i, :] = np.logical_and(lower_bounds[i, 0] < sims[0, :], sims[0, :] < upper_bounds[0, :])

    else:
        ha_normal_lower[here2, 0] = norm.ppf(target_betas[here2, 0]) * cov_matrix_ha[0, 0] ** 0.5 + means_ha[0]
        lower_bounds[here2, 0] = np.floor(ha_normal_lower[here2, 0]).astype(int)

    if mode != 'normal' and np.any(here2):
        c_p = vect_MW_CDF(lower_bounds[here2, 0], sample_sizes[0, 0], sample_sizes[0, 1])
        h0_normal_lower[here2, 0] = norm.ppf(c_p) * cov_matrix_h0[0, 0] ** 0.5 + means_h0[0]

    # endregion
    # evaluate models that are done
    done = lower_bounds[:, 0] + 1 >= upper_bounds[:, 0]
    lower_bounds[done, 0] = upper_bounds[done, 0] - 1

    if np.any(done):
        if mode == 'marginally exact':
            c_p = HA_CDF_approximation(np.array(lower_bounds[done, 0]).reshape(-1, 1), sample_sizes[0, 0].reshape(1),
                                       sample_sizes[0, 1].reshape(1), cohens_d, "Min ARE", max_rows=30)
            updated_betas[done, :] = np.tile(np.array(c_p).reshape((-1, 1)), (1, len(done)))
        elif mode == 'simulation':
            c_p = np.sum(upper_bounds[np.newaxis, done, 0] > sims[0, :, np.newaxis], axis=0) / n
            updated_betas[done, :] = 1 - np.tile(c_p.reshape((-1, 1)), (1, done.size))
        else:
            updated_betas[done, :] = norm.cdf((lower_bounds[done, 0] - means_ha[0])/cov_matrix_ha[0, 0] ** 0.5)

    not_done = lower_bounds[:, 0] + 1 < upper_bounds[:, 0]

    for i in np.arange(n_analyses):
        here = np.arange(n_models)[not_done][target_alphas[not_done, i] - updated_alphas[not_done, i - 1] > 10 ** -10]
        if here.size > 0:
            for j in here:
                def find_this_h0(u):
                    suggested = h0_normal_lower[j, :].copy()
                    suggested[i] = u
                    p = normal_probability(h0_normal_upper[j, :], suggested, True)
                    return target_alphas[j, i] - updated_alphas[j, i - 1] - p

                guess1 = norm.ppf(1 - target_alphas[j, i]) * cov_matrix_h0[i, i] ** 0.5 + means_h0[i]
                h0_normal_upper[j, i] = find_root(guess1, find_this_h0)

            if mode != 'normal':
                c_p = norm.cdf((h0_normal_upper[here, i] - means_h0[i])/cov_matrix_h0[i, i])
                mirror, qs = \
                    transform_h0(sample_sizes[i, 0], sample_sizes[i, 1], 1 - c_p,
                                 np.ceil(h0_normal_upper[here, i]).astype(int))
                upper_bounds[here, i] = np.prod(sample_sizes[i, :]) - mirror

                h0_normal_upper[here, i] = norm.ppf(1 - qs) * cov_matrix_h0[i, i] ** 0.5 + means_h0[i]
                for j in here:
                    lo = h0_normal_lower[j, :].copy()
                    lo[i] = h0_normal_upper[j, i]
                    up = h0_normal_upper[j, :].copy
                    up[i] = np.inf
                    updated_alphas[j, i] = updated_alphas[j, i - 1] + normal_probability(up, lo, True)
            else:
                upper_bounds[here, i] = np.ceil(h0_normal_upper[here, i]).astype(int)
                upper_bounds[upper_bounds < 0] = -1

        here2 = np.arange(n_models)[not_done][target_betas[not_done, i] - updated_betas[not_done, i - 1] > 10 ** -10]
        if here2.size > 0 and mode != 'simulation':
            for j in here2:
                def find_this_ha(u):
                    suggested = ha_normal_upper[j, :].copy()
                    suggested[i] = u
                    p = normal_probability(suggested, ha_normal_lower[j, :], False)
                    return target_betas[j, i] - updated_betas[j, i - 1] - p

                guess1 = norm.ppf(target_betas[j, i]) * cov_matrix_ha[i, i] ** 0.5 + means_ha[i]
                ha_normal_lower[j, i] = find_root(guess1, find_this_ha, f_increasing=False)

            if mode == 'marginally exact':
                c_p = norm.cdf((ha_normal_lower[here2, i] - means_ha[i]) / cov_matrix_ha[i, i])
                lower_bounds[here2, i], qs = \
                    transform_ha(n1=sample_sizes[i, 0], n2=sample_sizes[i, 1], marginal_betas=c_p,
                                 normal_guesses=np.floor(ha_normal_lower[here2, 0]).astype(int), cohens_d=cohens_d)

                ha_normal_lower[here, i] = norm.ppf(qs) * cov_matrix_ha[i, i] ** 0.5 + means_ha[i]
                for j in here:
                    lo = ha_normal_lower[j, :].copy()
                    lo[i] = -np.inf
                    up = ha_normal_upper[j, :].copy
                    up[i] = ha_normal_lower[j, i]
                    updated_betas[j, i] = updated_betas[j, i - 1] + normal_probability(up, lo, False)

                if here.size > 0:
                    c_p = HA_CDF_approximation(np.array(upper_bounds[here, i]).reshape(-1, 1) - 1,
                                               sample_sizes[i, 0].reshape(1),
                                               sample_sizes[i, 1].reshape(1), cohens_d, "Min ARE", max_rows=30)
                    ha_normal_upper[here, i] = norm.ppf(c_p) * cov_matrix_ha[i, i] ** 0.5 + means_ha[i]

            else:
                lower_bounds[here2, i] = np.floor(ha_normal_lower[here2, i]).astype(int)
                lower_bounds[lower_bounds > np.prod(sample_sizes[i, :])] = np.prod(sample_sizes[i, :])

        elif mode == 'simulation':
            for j in range(n_models):
                if 10 ** -10 < target_betas[j, i] - updated_betas[j, i - 1] < np.sum(undecided)/n:
                    lower_bounds[j, i] = np.round(np.quantile(np.array(sims[i, undecided]),
                                                              (target_betas[j, i] - updated_betas[j, i-1])
                                                              * n/np.sum(undecided))) - 1
                    if lower_bounds[j, i] + 1 >= upper_bounds[j, i]:
                        updated_betas[j, i] = np.sum(upper_bounds[j, i] - 1 >= sims[i, undecided]) / n
                    else:
                        updated_betas[j, i] = np.sum(lower_bounds[j, i] >= sims[i, undecided]) / n

                elif np.sum(undecided)/n <= target_betas[j, i] - updated_betas[j, i - 1]:
                    lower_bounds[j, i] = upper_bounds[j, i] - 1

                undecided[j, undecided[j, :]] = np.logical_and(lower_bounds[j, i] < sims[i, undecided[j, :]],
                                                               sims[i, undecided[j, :]] < upper_bounds[j, i])

        if mode != 'normal' and here2.size > 0:
            c_p = vect_MW_CDF(lower_bounds[here2, i], sample_sizes[i, 0], sample_sizes[i, 1])
            h0_normal_lower[here2, i] = norm.ppf(c_p) * cov_matrix_h0[i, i] ** 0.5 + means_h0[i]

        # evaluate models that are done
        new_done = np.logical_and(lower_bounds[:, i] + 1 >= upper_bounds[:, i], not_done)
        lower_bounds[new_done, i] = upper_bounds[new_done, i] - 1

        if np.any(new_done):
            if mode == 'marginally exact':
                c_p = HA_CDF_approximation(np.array(lower_bounds[new_done, i]).reshape(-1, 1),
                                           sample_sizes[i, 0].reshape(1),
                                           sample_sizes[i, 1].reshape(1), cohens_d, "Min ARE", max_rows=30)
                ha_normal_lower[new_done, i] = norm.cdf(c_p) * cov_matrix_ha[i, i] ** 0.5 + means_ha[i]

                for j in np.arange(n_models)[new_done]:
                    lo = ha_normal_lower[j, :].copy()
                    lo[i] = -np.inf
                    up = ha_normal_upper[j, :].copy
                    up[i] = ha_normal_lower[j, i]
                    updated_betas[j, i] = updated_betas[j, i - 1] + normal_probability(up, lo, False)

            elif mode == 'simulation':
                pass
            else:
                for j in np.arange(n_models)[new_done]:
                    lo = ha_normal_lower[j, :].copy()
                    lo[i] = -np.inf
                    up = ha_normal_upper[j, :].copy
                    up[i] = lower_bounds[j, i]
                    updated_betas[j, i] = updated_betas[j, i - 1] + normal_probability(up, lo, False)
            not_done[not_done] = lower_bounds[not_done, i] + 1 >= upper_bounds[not_done, i]

    return upper_bounds, lower_bounds, updated_betas

# region Normal not adapted


def simulate_statistics(n_simulations, sample_sizes, memory_limit, cohens_d, sides):
    """ Simulate test statistics for an independent groups t-test

    Simulate [param: n_simulations] test statistics for a t-test with effect size [param: cohens_d] and
    [param: sample_sizes]. [param: sides] determines if the test is treated as two-sided or one-sided.
    The memory limit makes sure to break the simulation up in smaller pieces such that the allocated memory is not
    exceeded too much."""

    n_simulations = int(n_simulations)
    sample_sizes = np.asarray(sample_sizes)
    n_analyses = int(sample_sizes.shape[1])
    sample_sizes.astype(int)
    Ts = np.zeros((n_analyses, n_simulations))

    sim_limit = int(np.floor(0.9 * (10 ** 9 / 8 * memory_limit - n_simulations * n_analyses) /
                             (1.5 * np.max(sample_sizes[:, -1]) + 5 * n_analyses)))
    repeats = int(np.ceil(n_simulations / sim_limit))
    sim_per_rep = int(np.ceil(n_simulations / repeats))
    dif = n_simulations - repeats * sim_per_rep
    sims = sim_per_rep

    for i in range(repeats):
        if i == repeats - 1:
            sims = sim_per_rep + dif

        x = np.random.normal(loc=0, size=sims * sample_sizes[0, -1])
        x = np.reshape(x, (sample_sizes[0, -1], sims))

        gr_mean1 = np.cumsum(x, axis=0)[sample_sizes[0, :] - 1, :] / sample_sizes[0, :, np.newaxis]
        SSE1 = np.zeros((n_analyses, sims))
        for j in range(n_analyses):
            SSE1[j, :] = np.sum((x[:sample_sizes[0, j], :] - gr_mean1[j, :]) ** 2, axis=0)

        x = np.random.normal(loc=cohens_d, size=sims * sample_sizes[1, -1])
        x = np.reshape(x, (sample_sizes[1, -1], sims))

        gr_mean2 = np.cumsum(x, axis=0)[sample_sizes[1, :] - 1, :] / sample_sizes[1, :, np.newaxis]
        SSE2 = np.zeros((n_analyses, sims))
        for j in range(n_analyses):
            SSE2[j, :] = np.sum((x[:sample_sizes[1, j], :] - gr_mean2[j, :]) ** 2, axis=0)

        Ts[:, (i * sim_per_rep):(i * sim_per_rep + sims)] = \
            (gr_mean2 - gr_mean1) / ((1 / sample_sizes[0, :, np.newaxis] + 1 / sample_sizes[1, :, np.newaxis]) *
                                     (SSE1 + SSE2) /
                                     (sample_sizes[0, :, np.newaxis] + sample_sizes[1, :, np.newaxis] - 2)) ** 0.5
        del gr_mean2, gr_mean1, SSE1, SSE2

    if sides == 'one':
        return Ts
    elif sides == 'two':
        return np.absolute(Ts)


def give_exact(sample_sizes, alphas, betas, cohens_d, sides):
    """ Give the properties of the first interim analysis for the independent groups t-test

    The returned properties are: critical values (significance and futility bounds),
    the probability of a true negative under H0, power, lower, and upper limit for the test statistic"""

    n_spending_scenarios = int(alphas.shape[0])
    non_central_param = cohens_d * (sample_sizes[0, 0] * sample_sizes[1, 0] /
                                    (sample_sizes[0, 0] + sample_sizes[1, 0])) ** 0.5
    degrees_freedom = sum(sample_sizes[:, 0]) - 2

    if sides == 'one':
        # These are simply the mathematical formulas
        sig_bounds = t.ppf(1 - alphas[:, 0], df=degrees_freedom)
        fut_bounds = nct.ppf(betas[:, 0], df=degrees_freedom, nc=non_central_param)
        fut_bounds[fut_bounds > sig_bounds] = sig_bounds[fut_bounds > sig_bounds]

        exact_true_neg = t.cdf(fut_bounds, df=degrees_freedom)
        exact_power = 1 - nct.cdf(sig_bounds, df=degrees_freedom, nc=non_central_param)

        return sig_bounds, fut_bounds, exact_true_neg, exact_power, -np.inf, np.inf

    elif sides == 'two':
        sig_bounds = t.ppf(1 - 0.5 * alphas[:, 0], df=degrees_freedom)
        fut_bounds = np.ones(n_spending_scenarios)
        """ For the two-sided version, the exact formula of the futility bound is not that simple to derive due to the 
        asymmetry of the non-central distribution.

        However, it can easily be found as the root of the function below, which is 1D so this process is fast enough 
        for me to be unwilling to check if I can find an exact formula. #DealWithIt """

        for i in range(n_spending_scenarios):
            def try_fut_bound(T):
                dif = nct.cdf(T, df=degrees_freedom, nc=non_central_param) \
                      - nct.cdf(-T, df=degrees_freedom, nc=non_central_param) \
                      - betas[i, 0]
                return dif

            fut_bounds[i] = root_scalar(try_fut_bound, bracket=[0, sig_bounds[i]], method='bisect').root

        fut_bounds[fut_bounds > sig_bounds] = sig_bounds[fut_bounds > sig_bounds]
        fut_bounds[np.isnan(fut_bounds)] = sig_bounds[np.isnan(fut_bounds)]

        exact_true_neg = 2 * t.cdf(fut_bounds, df=degrees_freedom) - 1
        exact_power = 1 - nct.cdf(sig_bounds, df=degrees_freedom, nc=non_central_param) + \
                      nct.cdf(-sig_bounds, df=degrees_freedom, nc=non_central_param)

        return sig_bounds, fut_bounds, exact_true_neg, exact_power, 0, np.inf


def give_fixed_sample_size(cohens_d, alpha, beta, sides):
    if sides == 'one':
        sides = 1
    else:
        sides = 2

    n = int(np.round(((norm.ppf(1 - alpha / sides) + norm.ppf(1 - beta, loc=cohens_d)) / cohens_d) ** 2))
    typeII = nct.cdf(t.ppf(1 - alpha / sides, df=2 * n - 2), df=2 * n - 2, nc=cohens_d * (n ** 2 / (2 * n)) ** 0.5)

    while typeII > beta:
        n = n + 1
        typeII = nct.cdf(t.ppf(1 - alpha / sides, df=2 * n - 2), df=2 * n - 2, nc=cohens_d * (n ** 2 / (2 * n)) ** 0.5)

    return n, typeII


def get_p_equivalent(x, N):
    return 1 - t.cdf(x, df=sum(N) - 2)

# endregion
# region help functions, no interaction with other files


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
    p_1 = vect_MW_CDF(normal_guesses, n1, n2)
    u_1 = normal_guesses.copy()
    p_2 = p_1.copy()
    p_2[np.abs(p_1 - marginal_alphas) < 10 ** -10] = np.inf

    u_1[p_1 > marginal_alphas + 10 ** -10] -= 1
    p_1[p_1 > marginal_alphas + 10 ** -10] = vect_MW_CDF(u_1[p_1 > marginal_alphas + 10 ** -10], n1, n2)
    p_2[p_1 < marginal_alphas - 10 ** -10] = vect_MW_CDF(u_1[p_1 < marginal_alphas - 10 ** -10] + 1, n1, n2)

    done = np.logical_or(np.abs(p_1 - marginal_alphas) < 10 ** -10,
                         np.logical_and(p_1 < marginal_alphas - 10 ** -10,
                                        p_2 > marginal_alphas + 10 ** -10))
    while not np.all(done):
        too_high = p_1 > marginal_alphas + 10 ** -10
        u_1[too_high] -= 1
        p_2[too_high] = p_1[too_high]
        p_1[too_high] = vect_MW_CDF(u_1[too_high], n1, n2)

        equal = np.abs(p_2 - marginal_alphas) < 10 ** -10
        u_1[equal] += 1
        p_1[equal] = p_2[equal]
        p_2[equal] = np.inf

        too_low = p_2 < marginal_alphas - 10 ** -10
        u_1[too_low] += 1
        p_1[too_low] = p_2[too_low]
        p_2[too_low] = vect_MW_CDF(u_1[too_low], n1, n2)

        done = np.logical_or(np.abs(p_1 - marginal_alphas) < 10 ** -10,
                             np.logical_and(p_1 < marginal_alphas + 10 ** -10,
                                            marginal_alphas < p_2 - 10 ** -10))
    return u_1, p_1


def transform_ha(n1, n2, marginal_betas, normal_guesses, cohens_d, tol=10**-5):
    # it is more efficient to check a few values at once, because of shared caching within the iteration
    n_guesses = 3
    guesses = np.ones((n_guesses, 1), dtype=int) * np.median(normal_guesses)
    guesses[0] -= 1
    guesses[2] += 1

    n_vals = marginal_betas.size
    result_crit = np.ones(n_vals)
    result_ps = np.ones(n_vals)

    results = np.array(check_TypeII(guesses, np.array(n1).reshape(1), np.array(n2).reshape(1),
                                    cohens_d, "Min ARE", marginal_betas, max_rows=30, solution="lower", tol=tol))
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
                result_crit[i] = guesses[np.abs(marginal_betas[i] - results) < tol * marginal_betas[i]][-1]
                result_ps[i] = results[np.abs(marginal_betas[i] - results) < tol * marginal_betas[i]][-1]

                too_high.remove(i)
            elif results[-1] <= marginal_betas[i] + tol * marginal_betas[i]:
                result_crit[i] = guesses[results <= marginal_betas[i] + tol * marginal_betas[i]][-1]
                result_ps[i] = HA_CDF_approximation(result_crit[i], np.array(n1).reshape(1), np.array(n2).reshape(1),
                                                    cohens_d, "Min ARE", max_rows=30, tol=tol)

                too_high.remove(i)
            elif np.any(results <= marginal_betas[i] + tol * marginal_betas[i]):
                result_crit[i] = guesses[results <= marginal_betas[i] + tol * marginal_betas[i]][-1]
                result_ps[i] = results[results <= marginal_betas[i] + tol * marginal_betas[i]][-1]

                too_high.remove(i)
    guess_h = guesses[0]
    while too_low:
        guesses -= n_guesses
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
                                                    cohens_d, "Min ARE", max_rows=30, tol=tol)

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
