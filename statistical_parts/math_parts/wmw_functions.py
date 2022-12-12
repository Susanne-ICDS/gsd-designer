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
    lower_bounds = - np.ones((n_models, n_analyses))

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

    # define helper functions
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

    # calculate 1st analysis bounds
    here = target_alphas[:, 0] > 10 ** -10
    h0_normal_upper[here, 0] = norm.ppf(1 - target_alphas[here, 0]) * cov_matrix_h0[0, 0] ** 0.5 + means_h0[0]

    if mode != 'normal':
        upper_bounds[here, 0], updated_alphas[here, 0] = transform_h0(0, target_alphas[here, 0],
                                                                      h0_normal_upper[here, 0])
        h0_normal_upper[here, 0] = norm.ppf(1 - updated_alphas[here, 0]) * cov_matrix_h0[0, 0] ** 0.5 + means_h0[0]
    else:
        upper_bounds[here, 0] = h0_normal_upper[here, 0]

    if updated_betas[0] > 10 ** -10:
        ha_normal_lower[0] = norm.ppf(target_betas[0]) * cov_matrix_ha[0, 0] ** 0.5 + means_ha[0]

        if mode == 'marginally exact':
            lower_bounds[0], updated_betas[0] = transform_ha(0, target_betas[0], ha_normal_lower[0])
            ha_normal_lower[0] = norm.ppf(updated_betas[0]) * cov_matrix_ha[0, 0] ** 0.5 + means_ha[0]

            c_p = HA_CDF_approximation(np.array([upper_bounds[0]]), sample_sizes[0, 0].reshape(1),
                                       sample_sizes[0, 1].reshape(1), cohens_d, "Min ARE", max_rows=30)
            ha_normal_upper[0] = norm.ppf(c_p) * cov_matrix_ha[0, 0] ** 0.5 + means_ha[0]
        elif mode == 'simulation':
            pass
        else:
            lower_bounds[0] = ha_normal_lower[0]

        if mode != 'normal':
            c_p = fixed_MW_CDF(int(np.floor(lower_bounds[0])), sample_sizes[0, 0], sample_sizes[0, 1])
            h0_normal_lower[0] = norm.ppf(c_p) * cov_matrix_h0[0, 0] ** 0.5 + means_h0[0]

    # new_normal = norm.ppf(new_beta) * cov_matrix_ha[analysis, analysis] ** 0.5 + means_ha[analysis]
    # c_p = fixed_MW_CDF(int(np.floor(lower_bounds[analysis])), sample_sizes[analysis, 0], sample_sizes[analysis, 1])
    # h0_normal_lower[analysis] = norm.ppf(c_p) * cov_matrix_h0[analysis, analysis] ** 0.5 + means_h0[analysis]
    # c_p = HA_CDF_approximation(np.array([up]), sample_sizes[analysis, 0].reshape(1),
    #                            sample_sizes[analysis, 1].reshape(1), cohens_d, pdf, max_rows=30)
    # ha_normal_upper[analysis] = norm.ppf(c_p) * cov_matrix_ha[analysis, analysis] ** 0.5 + means_ha[analysis]

    def MWW_determine_bounds(sample_sizes, target_alphas, target_betas, cohens_d, pdf='Min ARE', transform_fut=True,
                             prev_obtained_upper=None, prev_obtained_lower=None, as_guess=False, alt_update=True,
                             fut_conservative=True):


        if prev_obtained_lower is None:
            start = 1

        else:
            start2 = prev_obtained_upper.shape[0]
            lower_bounds[:start2] = prev_obtained_lower
            start = min(start, start2)

        if n_analyses == 1 or upper_bounds[0] <= np.floor(lower_bounds[0]) + 1:
            powered = upper_bounds[0] <= np.floor(lower_bounds[0]) + 1
            if as_guess:
                return upper_bounds, lower_bounds, powered
            lower_bounds[0] = upper_bounds[0] - 1
            return upper_bounds, lower_bounds, powered

        update_alt_normal(0)

        for i in np.arange(start, n_analyses, 1):
            if target_alphas[i] > updated_alphas[i - 1] + 10 ** -10:

                def find_this_h0(t):
                    if alt_update:
                        suggested = h0_normal_lower
                    else:
                        suggested = normal_lower.copy()
                    suggested[i] = t
                    p = probability(normal_upper, suggested, True)
                    return target_alphas[i] - updated_alphas[i - 1] - p

                guess1 = norm.ppf(1 - updated_alphas[i]) * cov_matrix_h0[i, i] ** 0.5 + means_h0[i]
                now = find_this_h0(guess1)
                flag = True
                okay = False
                if np.abs(now) < 10 ** -5:
                    new_u = guess1
                    flag = False
                    okay = True
                elif now < 10 ** -5:
                    guess2 = 2 * guess1 + 1
                    for _ in range(20):
                        now = find_this_h0(guess2)
                        if np.abs(now) < 10 ** -5:
                            new_u = guess1
                            flag = False
                            okay = True
                            break
                        elif now > 10 ** -5:
                            okay = True
                            break
                        guess1 = guess2
                        guess2 *= 2
                else:
                    guess2 = guess1 - 1
                    for _ in range(20):
                        now = find_this_h0(guess2)
                        if np.abs(now) < 10 ** -5:
                            new_u = guess1
                            flag = False
                            okay = True
                            break
                        elif now < -10 ** -5:
                            okay = True
                            break
                        new_dif = 2 * (guess1 - guess2)
                        guess1 = guess2
                        guess2 -= new_dif
                if okay:
                    if flag:
                        new_u = root_scalar(find_this_h0, bracket=[guess1, guess2]).root
                else:
                    if guess2 > 0:
                        new_u = np.inf
                    else:
                        new_u = -np.inf

                if new_u == - np.inf:
                    upper_bounds[i] = - 1
                elif new_u == np.inf:
                    pass
                else:
                    alpha_1D = 1 - norm.cdf((new_u - means_h0[i]) / cov_matrix_h0[i, i] ** 0.5)
                    upper_bounds[i], new_u = transform_h0(i, alpha_1D)
                    updated_alphas[i] = updated_alphas[i] - find_this_h0(new_u)
            else:
                new_u = np.inf

            if target_betas[i] > updated_betas[i - 1] + 10 ** -10:
                def find_this_ha(t):
                    if alt_update:
                        suggested = ha_normal_upper
                    else:
                        suggested = normal_upper.copy()
                    suggested[i] = t
                    p = probability(suggested, normal_lower, False)
                    return target_betas[i] - updated_betas[i - 1] - p

                guess1 = norm.ppf(updated_betas[i]) * cov_matrix_ha[i, i] ** 0.5 + means_ha[i]
                now = find_this_ha(guess1)
                okay = False
                flag = True
                if np.abs(now) < 10 ** -5:
                    new_l = guess1
                    flag = False
                    okay = True
                elif now > 0:
                    guess2 = 2 * guess1 + 1
                    for _ in range(20):
                        now = find_this_ha(guess2)
                        if np.abs(now) < 10 ** -5:
                            new_l = guess1
                            flag = False
                            okay = True
                        elif now < 0:
                            okay = True
                            break
                        guess1 = guess2
                        guess2 *= 2
                else:
                    guess2 = guess1 - 1
                    okay = False
                    for _ in range(20):
                        now = find_this_ha(guess2)
                        if np.abs(now) < 10 ** -5:
                            new_l = guess1
                            flag = False
                            okay = True
                        elif now > 0:
                            okay = True
                            break
                        new_dif = 2 * (guess1 - guess2)
                        guess1 = guess2
                        guess2 -= new_dif
                if okay:
                    if flag:
                        new_l = root_scalar(find_this_ha, bracket=[guess1, guess2]).root
                else:
                    if guess2 > 0:
                        new_l = np.inf
                    else:
                        new_l = -np.inf
                if transform_fut:
                    beta_1D = norm.cdf((new_l - means_ha[i]) / cov_matrix_ha[i, i] ** 0.5)
                    if new_l == -np.inf:
                        lower_bounds[i] = new_l = -1
                    elif new_l == np.inf:
                        lower_bounds[i] = new_l = prod_ss[i]
                    else:
                        lower_bounds[i], new_l = transform_ha(i, beta_1D, int(new_l))
                    updated_betas[i] = updated_betas[i] - find_this_ha(new_l)
                else:
                    lower_bounds[i] = new_l
            else:
                new_l = -np.inf

            normal_upper[i] = new_u
            normal_lower[i] = new_l

            if i == n_analyses - 1:
                powered = upper_bounds[i] <= lower_bounds[i] + 1
                if as_guess:
                    return upper_bounds, lower_bounds, powered
                lower_bounds[i] = upper_bounds[i] - 1
                return upper_bounds, lower_bounds, powered
            elif alt_update:
                update_alt_normal(i)

            if upper_bounds[i] <= lower_bounds[i] + 1:
                if as_guess:
                    return upper_bounds, lower_bounds, upper_bounds[i] <= prod_ss[i]
                lower_bounds[i] = upper_bounds[i] - 1
                return upper_bounds, lower_bounds, upper_bounds[i] <= prod_ss[i]

    return []

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


def transform_ha(n1, n2, marginal_betas, normal_guesses, cohens_d):
    # it is more efficient to check a few values at once, because of shared caching within the iteration
    guesses = np.ones((3, 1), dtype=int) * np.median(normal_guesses)
    guesses[0] -= 1
    guesses[2] += 1
    old_guess = None

    for _ in range(n1*n2 + 1):
        # evaluate guesses, if it is clear the solution is contained within the guesses, automatically continues
        # iterating to desired precision level
        results = check_TypeII(guesses, np.array(n1).reshape(1), np.array(n2).reshape(1),
                                cohens_d, "Min ARE", marginal_betas, max_rows=30, solution="lower")
            results = np.array(results)

            if np.any(results <= marginal_beta) and np.any(results > marginal_beta):
                # solution found
                new_lower = guesses[results <= marginal_beta][-1]
                new_beta = results[results <= marginal_beta][-1]
                return new_lower, new_beta
            elif results[0] <= marginal_beta:
                if old_guess is None or old_guess < guesses[0]:
                    # check larger guesses
                    old_guess = guesses[2]
                    guesses += 3
                else:
                    # turns out the solution was in the previous guesses. Now we still need to get the marginal beta at
                    # the desired precision level
                    new_lower = guesses[2]
                    new_beta = HA_CDF_approximation(new_lower, sample_sizes[analysis, 0].reshape(1),
                                                    sample_sizes[analysis, 1].reshape(1), cohens_d, "Min ARE",
                                                    max_rows=30)
                    return new_lower, new_beta
            else:
                if old_guess is None or old_guess > guesses[0]:
                    # check smaller guesses
                    old_guess = guesses[0]
                    guesses -= 3
                else:
                    # turns out the solution was in the previous guesses. Now we still need to get the marginal beta at
                    # the desired precision level
                    new_lower = old_guess
                    new_beta = HA_CDF_approximation(new_lower, sample_sizes[analysis, 0].reshape(1),
                                                    sample_sizes[analysis, 1].reshape(1), cohens_d, "Min ARE",
                                                    max_rows=30)
                    return new_lower, new_beta

# endregion
