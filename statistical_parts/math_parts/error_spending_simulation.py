import numpy as np
import pandas as pd
from scipy.stats import norm


def get_sim_nr(alphas, betas, rel_tolerance):
    """ Determine a suitable number of simulations per estimate of the properties of the design
    based on the alphas and betas spent, and on the relative tolerance for the allowed error. """

    alphas = np.asarray(alphas)
    betas = np.asarray(betas)

    dif_alphas = alphas[:, 1:] - alphas[:, :-1]
    dif_betas = betas[:, 1:] - betas[:, :-1]

    if np.any(dif_alphas > 10**-4):
        if np.any(dif_betas > 10**-4):
            return int(np.ceil(max(np.max(1 / dif_alphas[dif_alphas > 10**-4] * rel_tolerance ** -1 * 10),
                                   np.max(1 / dif_betas[dif_betas > 10**-4] * rel_tolerance ** -1 * 10))))
        else:
            return int(np.ceil(np.max(1 / dif_alphas[dif_alphas > 10**-4] * rel_tolerance ** -1 * 10)))
    elif np.any(dif_betas > 10**-4):
        return int(np.ceil(np.max(1 / dif_betas[dif_betas > 10 ** -4] * rel_tolerance ** -1 * 10)))

    # This situation is nonsense since all max allowed error is spent at the first analysis which is exact
    # anyway. But you never know what kind of stuff users will enter.
    return int(np.ceil(rel_tolerance))


def simulate_empirical_bounds(alphas, betas, exact_sig, exact_fut, simulator_h0, simulator_ha, costs,
                              n_simulations: int, n_repeats: int, exact_true_neg=None, exact_power=None):
    """  Simulate the bounds and properties of the specified group sequential design.

    :param alphas: (n_models x n_analyses) array-like with the type I error spent per analysis for each GSD
    :param betas: (n_models x n_analyses) array-like with the type II error spent per analysis for each GSD

    :param exact_sig: (n_models) array-like with the exact significance bound for the first analysis for each GSD
    :param exact_fut: (n_models) array-like with the exact futility bound for the first analysis for each GSD

    :param simulator_h0: function(int) simulates int test statistics under the null hypothesis
    :param simulator_ha: function(int) simulates int test statistics under the alternative hypothesis

    :param costs: (n_models x n_analyses) array-like with the costs per analysis for each GSD
    :param n_simulations: number of simulations for a single set of estimates
    :param n_repeats: number of estimates, number of times the simulation process is repeated

    :param exact_true_neg: the exact probability of a true negative at the first analysis for each GSD
    :param exact_power: (n_models) array-like with the exact power at the first analysis for each GSD

    :return: numpy arrays with the estimates per repeat,
    either (n_models x n_analyses x n_repeats) or (n_models x n_repeats),
    sig_bounds, fut_bounds, mean_cost_h0, mean_cost_ha, power, true_negatives

    """

    alphas = np.asarray(alphas)
    betas = np.asarray(betas)

    n_models = alphas.shape[0]
    n_analyses = alphas.shape[1]

    costs = np.asarray(costs)
    costs = costs.reshape(n_analyses)
    mean_cost_h0 = np.zeros([n_models, n_repeats])
    mean_cost_ha = np.zeros([n_models, n_repeats])

    exact_sig = np.asarray(exact_sig)
    sig_bounds = np.ones([n_models, n_analyses, n_repeats])
    sig_bounds[:, 0, :] = np.repeat(exact_sig, n_repeats).reshape(n_models, n_repeats)
    exact_fut = np.asarray(exact_fut)
    fut_bounds = np.ones([n_models, n_analyses, n_repeats])
    fut_bounds[:, 0, :] = np.repeat(exact_fut, n_repeats).reshape(n_models, n_repeats)

    power = np.zeros([n_models, n_analyses, n_repeats])
    true_negatives = np.zeros([n_models, n_analyses, n_repeats])

    for r in range(n_repeats):
        T0 = simulator_h0(n_simulations)
        TA = simulator_ha(n_simulations)

        for i in range(n_models):
            power[i, :, r] = np.sum(sig_bounds[i, 0, r] <= TA[0, :])
            true_negatives[i, :, r] = np.sum(fut_bounds[i, 0, r] >= T0[0, :])

            if sig_bounds[i, 0, r] <= fut_bounds[i, 0, r] + 10**-15:
                fut_bounds[i, 0, r] = sig_bounds[i, 0, r]

            undecided_h0 = np.logical_and(sig_bounds[i, 0, r] > T0[0, :], fut_bounds[i, 0, r] < T0[0, :])
            undecided_ha = np.logical_and(sig_bounds[i, 0, r] > TA[0, :], fut_bounds[i, 0, r] < TA[0, :])
            # Record at which analysis each simulation reached significance or was found futile
            stop_at_h0 = np.zeros(n_simulations, dtype=int)
            stop_at_ha = np.zeros(n_simulations, dtype=int)

            for j in np.arange(1, n_analyses, 1):
                # Number of undecided (not significant, not futile) simulations left
                left_h0 = np.sum(undecided_h0)
                left_ha = np.sum(undecided_ha)

                if sig_bounds[i, j - 1, r] <= fut_bounds[i, j - 1, r] + 10**-15:
                    # Set-up over-powered before last analysis
                    # The futility bound of the previous analysis was larger than the significance bound
                    sig_bounds[i, j:, r] = np.nan
                    fut_bounds[i, j:, r] = np.nan
                    break

                elif left_h0 / n_simulations <= alphas[i, j] - alphas[i, j - 1]:
                    # Set-up over-powered before last analysis, all remaining simulations are significant
                    # so we might as well have made them significant in the previous analysis
                    sig_bounds[i, j - 1, r] = fut_bounds[i, j - 1, r]
                    power[i, j:, r] = np.sum(sig_bounds[i, j - 1, r] <= TA[j - 1, undecided_ha]) + power[i, j:, r]
                    sig_bounds[i, j:, r] = np.nan
                    fut_bounds[i, j:, r] = np.nan
                    break

                # The counter goes up by one for the simulations that did not cross the critical values in the
                # previous analyses
                stop_at_h0 = stop_at_h0 + undecided_h0
                stop_at_ha = stop_at_ha + undecided_ha

                # Check if there is a meaningful amount of alpha spent in this analysis
                if alphas[i, j] - alphas[i, j - 1] > 10**-15:
                    sig_bounds[i, j, r] = np.quantile(T0[j, undecided_h0],
                                                      1 - ((alphas[i, j] - alphas[i, j - 1]) * n_simulations / left_h0))
                else:
                    sig_bounds[i, j, r] = np.inf

                if j == n_analyses - 1:
                    # If this is the last analysis, then futility = significance bound to force a decision
                    fut_bounds[i, j, r] = sig_bounds[i, j, r]

                elif left_ha / n_simulations <= (betas[i, j] - betas[i, j - 1]):
                    # set-up overpowered before final analyses, all remaining simulations are futile,
                    fut_bounds[i, j, r] = sig_bounds[i, j, r]

                    power[i, j:, r] += np.sum(sig_bounds[i, j, r] <= TA[j, undecided_ha])
                    true_negatives[i, j:, r] += np.sum(fut_bounds[i, j, r] >= T0[j, undecided_h0])

                    sig_bounds[i, j:] = np.nan
                    fut_bounds[i, j:] = np.nan
                    break

                elif betas[i, j] - betas[i, j - 1] > 10**-15:
                    fut_bounds[i, j, r] = np.quantile(TA[j, undecided_ha],
                                                   (betas[i, j] - betas[i, j - 1]) * n_simulations / left_ha)
                    if fut_bounds[i, j, r] >= sig_bounds[i, j, r]:
                        fut_bounds[i, j, r] = sig_bounds[i, j, r]
                else:
                    fut_bounds[i, j, r] = - np.inf

                # Add the number of times stopped for significance under HA and for futility under H0
                power[i, j:, r] += np.sum(sig_bounds[i, j, r] <= TA[j, undecided_ha])
                true_negatives[i, j:, r] += np.sum(fut_bounds[i, j, r] >= T0[j, undecided_h0])

                undecided_h0 = np.logical_and(np.logical_and(sig_bounds[i, j, r] > T0[j, :],
                                                             fut_bounds[i, j, r] < T0[j, :]), undecided_h0)
                undecided_ha = np.logical_and(np.logical_and(sig_bounds[i, j, r] > TA[j, :],
                                                             fut_bounds[i, j, r] < TA[j, :]), undecided_ha)

            mean_cost_h0[i, r] = np.nanmean(costs[stop_at_h0])
            mean_cost_ha[i, r] = np.nanmean(costs[stop_at_ha])

    power = power / n_simulations
    true_negatives = true_negatives / n_simulations

    if exact_true_neg is not None:
        true_negatives[:, 0, :] = exact_true_neg[:, np.newaxis]
    if exact_power is not None:
        power[:, 0, :] = exact_power[:, np.newaxis]

    return sig_bounds, fut_bounds, mean_cost_h0, mean_cost_ha, power, true_negatives


def simulation_loop(alphas, betas, exact_sig, exact_fut, rel_tol, CI, col_names, model_ids, default_n_repeats,
                    max_n_repeats, simulator_h0, simulator_ha, costs, exact_true_neg=None, exact_power=None):
    """ Simulate estimates for the properties for the GSDs until the desired relative tolerance level
    for the error confidence interval has been reached."""

    n_simulations = get_sim_nr(alphas, betas, rel_tol)
    n_models = alphas.shape[0]
    n_analyses = alphas.shape[1]
    n_repeats = default_n_repeats

    # Initially all models require simulations -> sims_needed all True
    sims_needed = np.ones(n_models, dtype='?')

    sims_df = pd.DataFrame(columns=col_names)
    counts = pd.DataFrame(columns=col_names)
    estimates = pd.DataFrame(columns=col_names)
    std_errors = pd.DataFrame(columns=col_names)

    model_ids = np.asarray(model_ids)

    def transform(self, n_reps, n_mods):
        """ Simulation is in 3D (n_models x n_analyses x n_repeats), dataframes only accept 2D.
        The results are therefore reshaped and labelled per model. """

        return self.transpose([1, 0, 2]).reshape(n_analyses, n_reps * sum(n_mods)).transpose()

    def label(included_models):
        """ Create labels to match above transformation. """
        return np.repeat(np.arange(0, n_models)[included_models], n_repeats)[:, np.newaxis]

    while np.any(sims_needed):
        sig_bounds, fut_bounds, mean_cost_h0, mean_cost_ha, power, true_negatives = \
            simulate_empirical_bounds(alphas[sims_needed, :], betas[sims_needed, :], exact_sig[sims_needed],
                                      exact_fut[sims_needed], simulator_h0, simulator_ha, costs, n_simulations,
                                      n_repeats, exact_true_neg[sims_needed], exact_power[sims_needed])

        sims_df = sims_df.append(pd.DataFrame(
            np.concatenate((label(sims_needed),
                            transform(sig_bounds, n_repeats, sims_needed),
                            transform(fut_bounds, n_repeats, sims_needed),
                            mean_cost_h0.reshape(sum(sims_needed) * n_repeats, 1),
                            mean_cost_ha.reshape(sum(sims_needed) * n_repeats, 1),
                            transform(power, n_repeats, sims_needed),
                            transform(true_negatives, n_repeats, sims_needed)), axis=1),
            columns=col_names))

        sims_df[col_names[1:]] = sims_df[col_names[1:]].apply(pd.to_numeric, errors='ignore')

        counts = sims_df.groupby('Model id', as_index=True).count()
        estimates = sims_df.groupby('Model id', as_index=True).mean()
        std_errors = sims_df.groupby('Model id', as_index=True).std() / (counts ** 0.5)

        # Z-score * SE = length of the CI
        rel_error = (abs(norm.ppf(0.5 * (1 - CI))) * std_errors / estimates).max(axis=1)
        ratio = np.asarray(rel_error / rel_tol)
        sims_needed = ratio > 1

        # Estimate remaining required simulations to use for next iteration
        # Minimum and maximum added simulations per iteration: _default_n_repeats and _max_n_repeats
        if np.any(sims_needed):
            count = (np.asarray(counts[sims_needed])[0, 0])
            n_repeats = int(np.min(np.ceil(count * ratio[sims_needed] ** 2 - count)))
            n_repeats = min(max_n_repeats, max(default_n_repeats, n_repeats))

    estimates['Model id'] = model_ids
    std_errors['Model id'] = model_ids
    counts['Model id'] = model_ids

    return estimates, std_errors, n_simulations, counts


def simulate_bound_properties(sig_bounds, fut_bounds, simulator, costs, n_simulations: int, n_repeats: int,
                              exact_fut_prop=None, exact_sig_prop=None):
    """  Simulate the properties of the specified group sequential design with given bounds.

    :param sig_bounds: (n_models x n_analyses) array-like with the critical values for significance for each GSD
    :param fut_bounds: (n_models x n_analyses) array-like with the critical values for futility for each GSD

    :param simulator: function(n: int) simulates n test statistics under a chosen hypothesis

    :param costs: (n_models x n_analyses) array-like with the costs per analysis for each GSD
    :param n_simulations: number of simulations for a single set of estimates
    :param n_repeats: number of estimates, number of times the simulation process is repeated

    :param exact_fut_prop: the exact probability of a true negative at the first analysis for each GSD
    :param exact_sig_prop: (n_models) array-like with the exact power at the first analysis for each GSD

    :return: numpy arrays with the estimates per repeat,
    either (n_models x n_repeats) or (n_models x n_analyses x n_repeats),
    mean_cost, proportion_sig, proportion_fut

    """

    n_models = sig_bounds.shape[0]
    n_analyses = sig_bounds.shape[1]

    costs = np.asarray(costs)
    costs = costs.reshape(n_analyses)
    mean_cost = np.zeros([n_models, n_repeats])

    proportion_sig = np.zeros([n_models, n_analyses, n_repeats])
    proportion_fut = np.zeros([n_models, n_analyses, n_repeats])

    for r in range(n_repeats):
        Ts = simulator(n_simulations)

        for i in range(n_models):
            # Record at which analysis each simulation reached significance or was found futile
            stop_at = -np.ones(n_simulations, dtype=int)
            undecided = np.ones(n_simulations, dtype=bool)

            for j in np.arange(0, n_analyses, 1):
                # Number of undecided (not significant, not futile) simulations left
                left = n_simulations - proportion_sig[i, j, r] - proportion_fut[i, j, r]

                if left == 0:
                    # Decision made for each simulation
                    break

                # The counter goes up by one for the simulations that did not cross the critical values in the
                # previous analyses
                stop_at = stop_at + undecided

                proportion_sig[i, j:, r] += np.sum(sig_bounds[i, j] <= Ts[j, undecided])
                proportion_fut[i, j:, r] += np.sum(fut_bounds[i, j] >= Ts[j, undecided])

                undecided = np.logical_and(np.logical_and(sig_bounds[i, j] > Ts[j, :], fut_bounds[i, j] < Ts[j, :]),
                                           undecided)

            mean_cost[i, r] = np.nanmean(costs[stop_at])

    proportion_sig = proportion_sig / n_simulations
    proportion_fut = proportion_fut / n_simulations

    if exact_fut_prop is not None:
        proportion_fut[:, 0, :] = exact_fut_prop[:, np.newaxis]
    if exact_sig_prop is not None:
        proportion_sig[:, 0, :] = exact_sig_prop[:, np.newaxis]

    return mean_cost, proportion_sig, proportion_fut


def simulate_prop_loop(sig_bounds, fut_bounds, rel_tol, CI, col_names, model_ids, default_n_repeats, max_n_repeats,
                       simulator, costs, exact_fut_prop=None, exact_sig_prop=None, n_simulations=10 ** 6):
    sig_bounds = np.asarray(sig_bounds)
    fut_bounds = np.asarray(fut_bounds)

    n_models = sig_bounds.shape[0]
    n_analyses = sig_bounds.shape[1]
    n_repeats = default_n_repeats

    # Initially all models require simulations -> sims_needed all True
    sims_needed = np.ones(n_models, dtype='?')

    sims_df = pd.DataFrame(columns=col_names)
    counts = pd.DataFrame(columns=col_names)
    estimates = pd.DataFrame(columns=col_names)
    std_errors = pd.DataFrame(columns=col_names)

    model_ids = np.asarray(model_ids)

    def transform(self, n_reps, n_mods):
        """ Simulation is in 3D (n_models x n_analyses x n_repeats), dataframes only accept 2D.
        The results are therefore reshaped and labelled per model. """

        return self.transpose([1, 0, 2]).reshape(n_analyses, n_reps * sum(n_mods)).transpose()

    def label(included_models):
        """ Create labels to match above transformation. """
        return np.repeat(np.arange(0, n_models)[included_models], n_repeats)[:, np.newaxis]

    while np.any(sims_needed):
        mean_cost, proportion_sig, proportion_fut = \
            simulate_bound_properties(sig_bounds[sims_needed, :], fut_bounds[sims_needed, :], simulator, costs,
                                      n_simulations, n_repeats, exact_fut_prop, exact_sig_prop)

        sims_df = sims_df.append(pd.DataFrame(
            np.concatenate((label(sims_needed),
                            mean_cost.reshape(sum(sims_needed) * n_repeats, 1),
                            transform(proportion_sig, n_repeats, sims_needed),
                            transform(proportion_fut, n_repeats, sims_needed)), axis=1),
            columns=col_names))

        sims_df[col_names[1:]] = sims_df[col_names[1:]].apply(pd.to_numeric, errors='ignore')

        counts = sims_df.groupby('Model id', as_index=True).count()
        estimates = sims_df.groupby('Model id', as_index=True).mean()
        std_errors = sims_df.groupby('Model id', as_index=True).std() / (counts ** 0.5)

        # Z-score * SE = length of the CI
        rel_error = (abs(norm.ppf(0.5 * (1 - CI))) * std_errors / estimates).max(axis=1)
        ratio = np.asarray(rel_error / rel_tol)
        sims_needed = ratio > 1

        # Estimate remaining required simulations to use for next iteration
        # Minimum and maximum added simulations per iteration: _default_n_repeats and _max_n_repeats
        if np.any(sims_needed):
            count = (np.asarray(counts[sims_needed])[0, 0])
            n_repeats = int(np.min(np.ceil(count * ratio[sims_needed] ** 2 - count)))
            n_repeats = min(max_n_repeats, max(default_n_repeats, n_repeats))

    estimates['Model id'] = model_ids
    std_errors['Model id'] = model_ids
    counts['Model id'] = model_ids

    return estimates, std_errors, n_simulations, counts


'''
# test lines
from statistical_parts.math_parts.t_test_functions import simulate_statistics

rel_tol = 0.01
CI=0.95
n_analyses = 3
model_ids = ['OBF']
col_names = ['Model id'] + ['Expected_cost'] + \
                ['% significant at analysis {}'.format(i + 1) for i in range(n_analyses)] + \
                ['% futile at analysis {}'.format(i + 1) for i in range(n_analyses)]
default_n_repeats = 10
max_n_repeats = 100

sig_bounds = [(6.592, 2.284, 1.757)]
sig_bounds = [(3.186315546, 2.332, 1.947)]
sig_bounds = [(2.959537434, 2.373, 1.966)]

fut_bounds = [(-0.08572, 1.286, 1.757)]
fut_bounds = [(0.2225545674, 1.177, 1.947)]
fut_bounds = [(0.3414654458, 1.1667, 1.966)]

ds = np.arange(1.1, 1.3, 0.001)

def simulator(n_sims):
        return simulate_statistics(n_sims, np.array([(3, 6, 9), (3, 6, 9)]), 4, cohens_d=ds[i], sides='one')
    costs = [(6, 12, 18)]

cost_results = np.zeros(ds.size)

for i in range(ds.size):

    estimates, std_errors, n_simulations, counts = \
        simulate_prop_loop(sig_bounds, fut_bounds, rel_tol, CI, col_names, model_ids, default_n_repeats, max_n_repeats,
                           simulator, costs, exact_fut_prop=None, exact_sig_prop=None, n_simulations=10 ** 5)
    
    cost_results[i] = estimates['Expected_cost']
    print(cost_results[i], ds[i])



np.max(cost_results)
12.844612000000001
12.463152
12.786176

d
1.24
1.16
1.1

'''
