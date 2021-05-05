import numpy as np
import pyximport
pyximport.install()

from statistical_parts.math_parts.t_test_functions import simulate_statistics


def simulate_statistics2(n_simulations=10**6, sample_sizes=np.array([(3, 6, 9, 24), (3, 8, 10, 12)]),
                         memory_limit=0.5, cohens_d=1.1, sides="one"):
    """ Simulate test statistics for an independent groups t-test

    Simulate [param: n_simulations] test statistics for a t-test with effect size [param: cohens_d] and
    [param: sample_sizes]. [param: sides] determines if the test is treated as two-sided or one-sided. """

    sample_sizes = np.asarray(sample_sizes)
    n_analyses = int(sample_sizes.shape[1])
    sample_sizes.astype(int)
    Ts = np.zeros((n_analyses, n_simulations))

    sim_limit = int(np.floor((10**9 / 8 * memory_limit - n_simulations*n_analyses) /
                             (np.max(sample_sizes[:, -1]) + 4 * n_analyses)))
    repeats = 3 * int(np.ceil(n_simulations/sim_limit))
    sim_per_rep = int(np.ceil(n_simulations/repeats))
    dif = n_simulations - repeats*sim_per_rep
    sims = sim_per_rep

    for i in range(repeats):
        if i == repeats-1:
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

        del x

        Ts[:, (i*sim_per_rep):(i*sim_per_rep + sims)] = \
            (gr_mean2 - gr_mean1) / ((1 / sample_sizes[0, :, np.newaxis] + 1 / sample_sizes[1, :, np.newaxis]) *
                                     (SSE1 + SSE2) /
                                     (sample_sizes[0, :, np.newaxis] + sample_sizes[1, :, np.newaxis] - 2)) ** 0.5
    return
    if sides == 'one':
        return Ts
    elif sides == 'two':
        return np.absolute(Ts)

