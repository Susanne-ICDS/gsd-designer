import numpy as np
from scipy.stats import norm
from scipy.stats import f
from scipy.stats import ncf

from statistical_parts.math_parts.error_spending_simulation import simulation_loop


def get_statistics(alphas, betas, sample_sizes, rel_tol, CI, col_names, model_ids, default_n_repeats,
                   max_n_repeats, costs, test_parameters, memory_limit):

    exact_sig, exact_fut, exact_true_neg, exact_power, lower_limit, upper_limit = \
        give_exact(sample_sizes, alphas, betas, **test_parameters)

    def simulator_h0(n_sim):
        return simulate_statistics(n_sim, sample_sizes, memory_limit, means=np.zeros(sample_sizes.shape[0]),
                                   sd=test_parameters['sd'])

    def simulator_ha(n_sim):
        return simulate_statistics(n_sim, sample_sizes, memory_limit, **test_parameters)

    estimates, std_errors, n_simulations, counts = simulation_loop(
        alphas, betas, exact_sig, exact_fut, rel_tol, CI, col_names, model_ids, default_n_repeats, max_n_repeats,
        simulator_h0, simulator_ha, costs, exact_true_neg, exact_power, lower_limit, upper_limit)

    counts_str = '{}'.format(counts[col_names[1]][0])
    for i in range(counts.shape[0] - 1):
        counts_str += ', ' + '{}'.format(counts[col_names[1]][i])

    return estimates, std_errors, 'Simulations finished: ', 'Results per model based on respectively ' + counts_str + \
        ' estimates with {} simulations each'.format(n_simulations)


def simulate_statistics(n_simulations, sample_sizes, memory_limit, means, sd):
    """ Simulate test statistics for a one-way ANOVA

    Simulate [param: n_simulations] test statistics for a t-test with group means [param: means] and standard
    deviation [param: sd]. """
    sample_sizes = np.asarray(sample_sizes).astype(int)
    total_sample_sizes = np.sum(sample_sizes, axis=0)

    n_analyses = int(sample_sizes.shape[1])
    n_groups = int(sample_sizes.shape[0])

    # Theoretical group means
    means = np.asarray(means).reshape(n_groups)

    sim_limit = int(np.floor(0.9 * (10 ** 9 / 8 * memory_limit - n_simulations * n_analyses) /
                             (2 * np.max(sample_sizes[:, -1]) + 4 * n_analyses)))
    repeats = int(np.ceil(n_simulations / sim_limit))
    sim_per_rep = int(np.ceil(n_simulations / repeats))
    dif = n_simulations - repeats * sim_per_rep
    sims = sim_per_rep

    Fs = np.zeros((n_analyses, n_simulations))
    for r in range(repeats):
        if r == repeats - 1:
            sims = sim_per_rep + dif

        # Object for storing sample means
        group_mean = np.zeros((sims, n_analyses, n_groups))
        within_group_var = np.zeros((sims, n_analyses))

        for i in range(n_groups):
            xs = np.random.normal(loc=means[i], scale=sd, size=(sims, sample_sizes[i, -1]))
            group_mean[:, :, i] = (np.cumsum(xs, axis=1)[:, sample_sizes[i, :] - 1] / sample_sizes[i, :])

            for j in range(n_analyses):
                within_group_var[:, j] += np.sum((group_mean[:, j, i][:, np.newaxis] - xs[:, :sample_sizes[i, j]]) ** 2,
                                                 axis=1)
            del xs

        group_mean = group_mean.transpose()
        within_group_var = within_group_var.transpose() / (total_sample_sizes[:, np.newaxis] - n_groups)

        grand_mean = np.sum(group_mean * sample_sizes[:, :, np.newaxis], axis=0) / \
            total_sample_sizes[:, np.newaxis]

        between_group_var = np.sum(sample_sizes[:, :, np.newaxis] *
                                   (group_mean - grand_mean[np.newaxis, :, :]) ** 2, axis=0) / (n_groups - 1)

        del grand_mean, group_mean

        Fs[:,  (r * sim_per_rep):(r * sim_per_rep + sims)] = between_group_var/within_group_var

        del between_group_var, within_group_var

    return Fs


def give_exact(sample_sizes, alphas, betas, means, sd):
    """ Give the properties of the first interim analysis for the independent groups t-test

    The returned properties are: critical values (significance and futility bounds),
    the probability of a true negative under H0 and power"""

    sample_sizes = np.asarray(sample_sizes)
    expected_means = np.asarray(means)
    n_groups = expected_means.size
    expected_means = expected_means.reshape(n_groups)

    grand_mean = np.sum(expected_means * sample_sizes[:, 0])/np.sum(sample_sizes[:, 0])
    non_central_param = np.sum(sample_sizes[:, 0] * (expected_means - grand_mean) ** 2)/sd**2
    denom_degrees_freedom = sum(sample_sizes[:, 0])-n_groups

    sig_bounds = f.ppf(1 - alphas[:, 0], dfn=n_groups-1, dfd=denom_degrees_freedom)
    fut_bounds = ncf.ppf(betas[:, 0], dfn=n_groups-1, dfd=denom_degrees_freedom, nc=non_central_param)
    fut_bounds[fut_bounds > sig_bounds] = sig_bounds[fut_bounds > sig_bounds]
    fut_bounds[np.isnan(fut_bounds)] = sig_bounds[np.isnan(fut_bounds)]

    exact_true_neg = f.cdf(fut_bounds, dfn=n_groups-1, dfd=denom_degrees_freedom)
    exact_power = 1 - ncf.cdf(sig_bounds, dfn=n_groups-1, dfd=denom_degrees_freedom, nc=non_central_param)
    exact_power[np.isnan(exact_power)] = 1 - 10**-8

    return sig_bounds, fut_bounds, exact_true_neg, exact_power, 0, np.inf


def give_fixed_sample_size(means, sd, alpha, beta):
    expected_means = np.asarray(means)
    n_groups = expected_means.size
    expected_means = expected_means.reshape(n_groups)

    grand_mean = np.sum(expected_means) / n_groups
    n = 3

    non_central_param = np.sum(n * (expected_means - grand_mean) ** 2) / sd ** 2
    typeII = ncf.cdf(f.ppf(1 - alpha, dfn=n_groups-1, dfd=n_groups*(n-1)), nc=non_central_param,
                     dfn=n_groups-1, dfd=n_groups*(n-1))

    while typeII > beta:
        n = n + 1
        non_central_param = np.sum(n * (expected_means - grand_mean) ** 2) / sd ** 2
        typeII = ncf.cdf(f.ppf(1 - alpha, dfn=n_groups - 1, dfd=n_groups * (n - 1)), nc=non_central_param,
                         dfn=n_groups - 1, dfd=n_groups * (n - 1))

    return n, typeII


def get_p_equivalent(x, N):
    n_groups = N.size
    return 1-f.cdf(x, dfn=n_groups - 1, dfd=sum(N) - n_groups)
