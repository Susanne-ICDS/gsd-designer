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
    """  Simulate the properties of the specified group sequential design.

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
            sigs = sig_bounds[i, 0, r] <= TA[0, :]
            futs = fut_bounds[i, 0, r] >= T0[0, :]

            power[i, :, r] = np.sum(sigs)
            true_negatives[i, :, r] = np.sum(futs)

            undecided_h0 = np.logical_and(sig_bounds[i, 0, r] > T0[0, :], np.logical_not(futs))
            undecided_ha = np.logical_and(np.logical_not(sigs), fut_bounds[i, 0, r] < TA[0, :])
            # Record at which analysis each simulation reached significance or was found futile
            stop_at_h0 = np.zeros(n_simulations, dtype=int)
            stop_at_ha = np.zeros(n_simulations, dtype=int)

            for j in np.arange(1, n_analyses, 1):
                # Number of undecided (not significant, not futile) simulations left
                left_h0 = np.sum(undecided_h0)
                left_ha = np.sum(undecided_ha)

                if sig_bounds[i, j - 1, r] <= fut_bounds[i, j - 1, r]:
                    # Set-up over-powered before last analysis
                    # The futility bound of the previous analysis was larger than the significance bound
                    fut_bounds[i, j - 1, r] = sig_bounds[i, j - 1, r]
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

                    power[i, j:, r] = np.sum(sig_bounds[i, j, r] <= TA[j, undecided_ha]) + power[i, j:, r]
                    true_negatives[i, j:, r] = np.sum(fut_bounds[i, j] >= T0[j, undecided_h0]) + true_negatives[i, j:]

                    sig_bounds[i, j:] = np.nan
                    fut_bounds[i, j:] = np.nan
                    break

                elif betas[i, j] - betas[i, j - 1] > 10**-15:
                    fut_bounds[i, j] = np.quantile(TA[j, undecided_ha],
                                                   (betas[i, j] - betas[i, j - 1]) * n_simulations / left_ha)
                else:
                    fut_bounds[i, j] = - np.inf

                # Add the number of times stopped for significance under HA and for futility under H0
                power[i, j:, r] = np.sum(sig_bounds[i, j, r] <= TA[j, undecided_ha]) + power[i, j:, r]
                true_negatives[i, j:, r] = np.sum(fut_bounds[i, j, r] >= T0[j, undecided_h0]) + true_negatives[i, j:, r]

                undecided_h0 = np.logical_and(sig_bounds[i, j, r] > T0[j, :],
                                              fut_bounds[i, j, r] < T0[j, :], undecided_h0)
                undecided_ha = np.logical_and(sig_bounds[i, j, r] > TA[j, :],
                                              fut_bounds[i, j, r] < TA[j, :], undecided_ha)

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

    # The 'group by' action orders models alphabetically,
    # The below index is used to make sure the correct models are identified as needing more simulations
    order = np.argsort(model_ids)
    model_ids = np.asarray(model_ids)

    def transform(self, n_reps, n_mods):
        """ Simulation is in 3D (n_models x n_analyses x n_repeats), dataframes only accept 2D.
        The results are therefore reshaped and labelled per model. """

        return self.transpose([1, 0, 2]).reshape(n_analyses, n_reps * sum(n_mods)).transpose()

    def label(included_models):
        """ Create labels to match above transformation. """
        return np.repeat(model_ids[included_models], n_repeats)[:, np.newaxis]

    while np.any(sims_needed):
        sig_bounds, fut_bounds, mean_cost_h0, mean_cost_ha, power, true_negatives = \
            simulate_empirical_bounds(alphas[sims_needed, :], betas[sims_needed, :], exact_sig[sims_needed],
                                      exact_fut[sims_needed], simulator_h0, simulator_ha, costs, n_simulations,
                                      n_repeats, exact_true_neg, exact_power)

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
        too_big = ratio > 1

        # Estimate remaining required simulations to use for next iteration
        # Minimum and maximum added simulations per iteration: _default_n_repeats and _max_n_repeats
        if np.any(too_big):
            count = (np.asarray(counts[too_big])[0, 0])
            n_repeats = int(np.min(np.ceil(count * ratio[too_big] ** 2 - count)))
            n_repeats = min(max_n_repeats, max(default_n_repeats, n_repeats))

        sims_needed[order] = too_big

    estimates['Model id'] = np.sort(model_ids)
    std_errors['Model id'] = np.sort(model_ids)
    counts['Model id'] = np.sort(model_ids)

    return estimates, std_errors, n_simulations, counts


def simulation_loop2(alphas, betas, exact_sig, exact_fut, rel_tol, CI, col_names, model_ids, default_n_repeats,
                     max_n_repeats, simulator_h0, simulator_ha, costs, exact_true_neg=None, exact_power=None,
                     max_iter=10**3):
    """ Simulate estimates for the properties for the GSDs until the desired relative tolerance level
    for the error confidence interval has been reached."""

    n_simulations = get_sim_nr(alphas, betas, rel_tol)
    n_models = alphas.shape[0]
    n_analyses = alphas.shape[1]
    n_repeats = default_n_repeats

    # Initially all models require simulations -> sims_needed all True
    sims_needed = np.ones(n_models, dtype='?')
    count = 0
    model_ids = np.asarray(model_ids)

    simulations = np.zeros((n_models, 4 * n_analyses + 2, 0))

    for _ in range(max_iter):
        sig_bounds, fut_bounds, mean_cost_h0, mean_cost_ha, power, true_negatives = \
            simulate_empirical_bounds(alphas[sims_needed, :], betas[sims_needed, :], exact_sig[sims_needed],
                                      exact_fut[sims_needed], simulator_h0, simulator_ha, costs, n_simulations,
                                      n_repeats, exact_true_neg, exact_power)

        add_this = np.nan + np.zeros((n_models, 4 * n_analyses + 2, n_repeats))
        add_this[sims_needed] = np.concatenate((sig_bounds, fut_bounds, mean_cost_h0[:, np.newaxis, :],
                                                mean_cost_ha[:, np.newaxis, :], power, true_negatives), axis=1)
        simulations = np.concatenate((simulations, add_this), axis=2)
        count = count + n_repeats

        estimates = np.mean(simulations[sims_needed, :, :], axis=2)
        sds = np.std(simulations[sims_needed, :, :], axis=2)
        # Z-score * SE = length of the CI
        rel_error = (abs(norm.ppf(0.5 * (1 - CI))) * sds / (count ** 0.5) / estimates).max(axis=1)
        ratio = np.asarray(rel_error / rel_tol)
        too_big = ratio > 1
        sims_needed[sims_needed] = too_big

        # Estimate remaining required simulations to use for next iteration
        # Minimum and maximum added simulations per iteration: _default_n_repeats and _max_n_repeats
        if np.any(too_big):
            n_repeats = int(np.min(np.ceil(count[sims_needed] * ratio[too_big] ** 2 - count[sims_needed])))
            n_repeats = min(max_n_repeats, max(default_n_repeats, n_repeats))
        else:
            break

    simulations = simulations.transpose((0, 2, 1)).reshape(n_models * count, 4 * n_analyses + 2)
    redundant_rows = np.all(np.isnan(simulations), axis=1)
    # Add model ids
    simulations = np.append(np.repeat(model_ids, count)[:, np.newaxis], simulations, axis=1)
    simulations = simulations[np.logical_not(redundant_rows), :]

    sims_df = pd.DataFrame(simulations, columns=col_names)
    counts = sims_df.groupby('Model id', as_index=True).count()
    estimates = sims_df.groupby('Model id', as_index=True).mean()
    std_errors = sims_df.groupby('Model id', as_index=True).std() / (counts ** 0.5)

    return estimates, std_errors, n_simulations, count
