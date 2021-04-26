import numpy as np
from scipy.stats import f
from scipy.stats import ncf


def simulate_statistics(n_simulations, sample_sizes, means, sd):
    """ Simulate test statistics for a one-way ANOVA

    Simulate [param: n_simulations] test statistics for a t-test with group means [param: means] and standard
    deviation [param: sd]. """
    sample_sizes = np.asarray(sample_sizes).astype(int)

    n_analyses = int(sample_sizes.shape[1])
    n_groups = int(sample_sizes.shape[0])

    # Theoretical group means
    means = np.asarray(means).reshape(n_groups)

    # Object for storing sample means
    group_mean = np.zeros((n_simulations, n_analyses, n_groups))
    within_group_SS = np.zeros((n_simulations, n_analyses, n_groups))

    for i in range(n_groups):
        xs = np.random.normal(loc=means[i], scale=sd, size=(n_simulations, sample_sizes[i, -1]))
        group_mean[:, :, i] = (np.cumsum(xs, axis=1)[:, sample_sizes[i, :] - 1] / sample_sizes[i, :])

        for j in range(n_analyses):
            within_group_SS[:, j, i] = np.sum((group_mean[:, j, i][:, np.newaxis] - xs[:, :sample_sizes[i, j]]) ** 2,
                                              axis=1)
    sample_sizes = sample_sizes.transpose()[np.newaxis, :]
    total_sample_sizes = np.sum(sample_sizes, axis=2)
    grand_mean = np.sum(group_mean * sample_sizes, axis=2) / total_sample_sizes

    within_group_var = np.sum(within_group_SS, axis=2) / (total_sample_sizes - n_groups)
    between_group_var = np.sum(total_sample_sizes[:, :, np.newaxis] *
                               (group_mean - grand_mean[:, :, np.newaxis]) ** 2, axis=2) / (n_groups - 1)

    Fs = between_group_var/within_group_var
    return Fs.transpose()


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

    exact_true_neg = f.cdf(fut_bounds, dfn=n_groups-1, dfd=denom_degrees_freedom)
    exact_power = 1 - ncf.cdf(sig_bounds, dfn=n_groups-1, dfd=denom_degrees_freedom, nc=non_central_param)

    return sig_bounds, fut_bounds, exact_true_neg, exact_power
