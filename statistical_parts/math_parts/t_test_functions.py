import numpy as np
from scipy.stats import norm
from scipy.stats import t
from scipy.stats import nct
from scipy.optimize import root_scalar


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
    the probability of a true negative under H0 and power"""

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

        return sig_bounds, fut_bounds, exact_true_neg, exact_power

    elif sides == 'two':
        sig_bounds = t.ppf(1 - 0.5 * alphas[:, 0], df=degrees_freedom)
        fut_bounds = np.ones(n_spending_scenarios)
        """ For the two-sided version, the exact formula of the futility bound is not that simple to derive due to the 
        asymmetry of the non-central distribution.
        
        However, it can easily be found as the root of the function below, which is 1D so this process is fast enough 
        for me to be unwilling to check if I can find an exact formula. #DealWithIt """

        for i in range(n_spending_scenarios):
            def try_fut_bound(T):
                dif = nct.cdf(T, df=degrees_freedom, nc=non_central_param)\
                   - nct.cdf(-T, df=degrees_freedom, nc=non_central_param)\
                   - betas[i, 0]
                return dif
            fut_bounds[i] = root_scalar(try_fut_bound, bracket=[0, sig_bounds[i]], method='bisect').root

        fut_bounds[fut_bounds > sig_bounds] = sig_bounds[fut_bounds > sig_bounds]
        fut_bounds[np.isnan(fut_bounds)] = sig_bounds[np.isnan(fut_bounds)]

        exact_true_neg = 2 * t.cdf(fut_bounds, df=degrees_freedom) - 1
        exact_power = 1 - nct.cdf(sig_bounds, df=degrees_freedom, nc=non_central_param) + \
            nct.cdf(-sig_bounds, df=degrees_freedom, nc=non_central_param)

        return sig_bounds, fut_bounds, exact_true_neg, exact_power


def give_fixed_sample_size(cohens_d, alpha, beta, sides):
    if sides == 'one':
        sides = 1
    else:
        sides = 2

    n = int(np.round(((norm.ppf(1 - alpha/sides) + norm.ppf(1-beta, loc=cohens_d))/cohens_d)**2))
    typeII = nct.cdf(t.ppf(1 - alpha/sides, df=2*n-2), df=2*n-2, nc=cohens_d * (n**2 / (2*n)) ** 0.5)

    while typeII > beta:
        n = n + 1
        typeII = nct.cdf(t.ppf(1 - alpha / sides, df=2 * n - 2), df=2 * n - 2, nc=cohens_d * (n ** 2 / (2 * n)) ** 0.5)

    return n, typeII


def get_p_equivalent(x, N):
    return 1-t.cdf(x, df=sum(N) - 2)

