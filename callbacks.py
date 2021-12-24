import dash
import dash_table
import dash_html_components as html
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State, MATCH, ALL
from dash_extensions.snippets import send_data_frame, send_bytes
import dash_bootstrap_components as dbc

import pandas as pd
import numpy as np
from scipy.stats import norm
import xlsxwriter

from app import dash_app

from statistical_parts.statistical_test_objects import TestObject
from statistical_parts.error_spending import check_form_error_spent
from statistical_parts.math_parts.error_spending_simulation import simulation_loop
from statistical_parts.math_parts.confidence_intervals import simulate_effect_CI

from layout_instructions import table_style, disabled_style_header, disabled_style_data, label
from layout_instructions import spacing_variables as spacing

_default_n_repeats = 10
_max_n_repeats = 10 ** 6


@dash_app.callback(
    Output('basic-design', 'hidden'),
    Output('interim-analyses', 'hidden'),
    Output('error-spending', 'hidden'),
    Output('simulation', 'hidden'),
    Output('effect-size-CI', 'hidden'),
    Output('tab1', 'active'),
    Output('tab2', 'active'),
    Output('tab3', 'active'),
    Output('tab4', 'active'),
    Output('tab5', 'active'),
    Output('previous', 'disabled'),
    Output('next', 'disabled'),
    Output('previous', 'href'),
    Output('next', 'href'),
    Input('url', 'pathname'))
def navigation(path):
    tabs = np.array(['/basic-design', '/interim-analyses', '/error-spending', '/simulation', '/effect-size-CI'])
    n_tabs = tabs.size

    if path is None:
        hidden_pages = np.ones(np.size(tabs), dtype=bool)
        hidden_pages[0] = False
        list_of_outputs = [None for _ in range(2 * n_tabs + 4)]
        list_of_outputs[:n_tabs] = hidden_pages
        list_of_outputs[n_tabs:2*n_tabs] = np.logical_not(hidden_pages)
        list_of_outputs[-4:] = [True, False, None, tabs[1]]
        return tuple(list_of_outputs)

    first = path == '/basic-design'
    last = path == '/effect-size-CI'
    current_tab = int(np.arange(0, np.size(tabs))[tabs == path])
    previous_tab = None
    next_tab = None

    if current_tab > 0:
        previous_tab = tabs[current_tab - 1]
    if current_tab < np.size(tabs) - 1:
        next_tab = tabs[current_tab + 1]

    return path != '/basic-design', path != '/interim-analyses', path != '/error-spending', path != '/simulation', \
        path != '/effect-size-CI', path == '/basic-design', path == '/interim-analyses', path == '/error-spending', \
        path == '/simulation', path == '/effect-size-CI', first, last, previous_tab, next_tab


@dash_app.callback(
    Output('test_input_tab1', 'children'),
    Input('stat_test', 'value'))
def display_tab1(stat_test):
    return TestObject(stat_test).tab_basic_design()


@dash_app.callback(
    Output('test_input_tab2', 'children'),
    Input('stat_test', 'value'))
def display_tab2(stat_test):
    return TestObject(stat_test).tab_interim_analyses()


@dash_app.callback(Output('explain-accuracy', 'children'),
                   Input('relative-tolerance', 'value'),
                   Input('CI', 'value'))
def explain_accuracy(rel_tol, CI):
    if rel_tol is None or CI is None:
        raise PreventUpdate

    return 'Simulations will continue until the {}%-confidence '.format(round(CI*100, ndigits=2)) + \
           'interval has a radius of less than {}% of the estimate. '.format(round(rel_tol*100, ndigits=2)) + \
           'The relative tolerance level determines to how many significant figures the results are rounded.'


# region test input
@dash_app.callback(
    Output('costs', 'style_data'),
    Output('costs', 'style_header'),
    Output('costs', 'editable'),
    Input('cost-default', 'value'))
def disable_costs(checked):
    if checked == ['default']:
        return disabled_style_data, disabled_style_header, False
    else:
        return table_style['style_data'], table_style['style_header'], True


@dash_app.callback(
    Output('costs', 'columns'),
    Output('costs', 'data'),
    Input('cost-default', 'value'),
    Input('sample_sizes', 'data'),
    Input('n_analyses', 'value'),
    Input('n_groups', 'value'),
    State('stat_test', 'value'))
def change_costs(checked, sample_sizes, n_analyses, n_groups, stat_test):
    """ Adjust the costs table to the number of analyses.
    If default is checked: update the costs based on the sample sizes """

    if sample_sizes is None or (n_analyses is None or n_groups is None):
        raise PreventUpdate

    cols = [{'name': 'Analysis {}'.format(i + 1), 'id': 'analysis-{}'.format(i), 'type': 'numeric'}
            for i in range(n_analyses)]

    # Check that none of the relevant cells is empty
    if np.any([sample_sizes[j]['analysis-{}'.format(i)] == '' for i in range(n_analyses) for j in range(n_groups)]):
        return cols, dash.no_update

    if checked == ['default']:
        return cols, TestObject(stat_test).default_costs(n_analyses, n_groups, sample_sizes)
    else:
        return cols, dash.no_update


@dash_app.callback(
    Output('sample_sizes', 'columns'),
    Input('n_analyses', 'value'))
def resize_columns(n_analyses):
    """ Adjust the size of the sample size input table to the number of analyses """
    if n_analyses is None:
        raise PreventUpdate

    return [{'name': 'Analysis {}'.format(i + 1), 'id': 'analysis-{}'.format(i), 'type': 'numeric'}
            for i in range(n_analyses)]


@dash_app.callback(
    Output('sample_sizes', 'data'),
    Input('n_groups', 'value'),
    State('sample_sizes', 'data'),
    State('stat_test', 'value'))
def resize_rows(n_groups, rows, stat_test):
    """ Adjust the size of the sample size input table to the number of groups """

    if rows is None or n_groups is None:
        raise PreventUpdate
    else:
        return TestObject(stat_test).resize_rows(n_groups, rows)


@dash_app.callback(
    Output({'type': 'test parameter', 'name': MATCH, 'form': 'datatable'}, 'data'),
    Input('n_groups', 'value'),
    State('stat_test', 'value'),
    State({'type': 'test parameter', 'name': MATCH, 'form': 'datatable'}, 'data'))
def update_input_table(n_groups, stat_test, rows):
    """ Adjust the test specific parameter input if necessary """

    if rows is not None and n_groups is not None:
        return TestObject(stat_test).update_input_table(n_groups=n_groups, rows=rows)

    raise PreventUpdate


@dash_app.callback(
    Output('fixed-n', 'is_open'),
    Output('fixed-n', 'color'),
    Output('fixed-n', 'children'),
    Input('url', 'pathname'),
    State('stat_test', 'value'),
    State('alpha', 'value'),
    State('beta', 'value'),
    State('n_groups', 'value'),
    State({'type': 'test parameter', 'name': ALL, 'form': 'value'}, 'value'),
    State({'type': 'test parameter', 'name': ALL, 'form': 'value'}, 'id'),
    State({'type': 'test parameter', 'name': ALL, 'form': 'datatable'}, 'data'),
    State({'type': 'test parameter', 'name': ALL, 'form': 'datatable'}, 'id')
)
def give_n(path, stat_test, alpha, beta, n_groups, test_param_values, test_param_values_ids, test_param_data,
           test_param_data_ids):
    if path == '/interim-analyses':
        color, message = TestObject(stat_test).fixed_sample_size(alpha, beta, test_param_values, test_param_values_ids,
                                                                 test_param_data, test_param_data_ids,
                                                                 n_groups=n_groups)
        return True, color, message
    raise PreventUpdate
# endregion


# region error-spending input
@dash_app.callback(
    Output('IR', 'hidden'),
    Output('DES', 'hidden'),
    Input('error_type', 'value'))
def switch_type(error_type):
    """ Display the chosen input type, hide the other """

    return 'IR' != error_type, 'DES' != error_type


@dash_app.callback(
    Output('information-ratio-table', 'style_data'),
    Output('information-ratio-table', 'style_header'),
    Output('information-ratio-table', 'editable'),
    Input('ir-default', 'value'))
def make_editable(checked):
    """ Make information input table editable if user does not want default values """

    if checked == ['default']:
        return disabled_style_data, disabled_style_header, False
    else:
        return table_style['style_data'], table_style['style_header'], True


@dash_app.callback(
    Output('information-ratio-table', 'columns'),
    Output('information-ratio-table', 'data'),
    Input('ir-default', 'value'),
    Input('sample_sizes', 'data'),
    Input('n_analyses', 'value'),
    Input('n_groups', 'value'),
    State('stat_test', 'value'))
def change_IR(checked, sample_sizes, n_analyses, n_groups, stat_test):
    """ Resize information ratio input table to match number of analyses.
    If default option is checked, update content as well. """

    if sample_sizes is None or n_analyses is None or n_groups is None:
        raise PreventUpdate

    cols = [{'name': 'Analysis {}'.format(i + 1), 'id': 'analysis-{}'.format(i), 'type': 'numeric'}
            for i in range(n_analyses)]

    # Check that none of the relevant cells is empty
    if np.any([sample_sizes[j]['analysis-{}'.format(i)] == '' for i in range(n_analyses) for j in range(n_groups)]):
        return cols, dash.no_update

    if checked == ['default']:
        return cols, TestObject(stat_test).default_information_ratio(n_analyses, n_groups, sample_sizes)
    else:
        return cols, dash.no_update


@dash_app.callback(
    Output('error-spending-table', 'columns'),
    Input('n_analyses', 'value'),
    Input('spending', 'value'))
def resize_table(n_analyses, spending):
    """ Adjust the size of the error spending input table to the number of analyses """

    if n_analyses is None:
        raise PreventUpdate

    alphas = [{'name': 'Alpha {}'.format(i + 1), 'id': 'alpha-{}'.format(i), 'type': 'numeric'}
              for i in range(n_analyses)]
    betas = [{'name': 'Beta {}'.format(i + 1), 'id': 'beta-{}'.format(i), 'type': 'numeric'}
             for i in range(n_analyses)]

    if spending == 'both':
        return alphas + betas

    if spending == 'alpha':
        return alphas

    if spending == 'beta':
        return betas


@dash_app.callback(
    Output('error-spending-table', 'data'),
    Input('adding-rows-button', 'n_clicks'),
    Input('n_analyses', 'value'),
    State('error-spending-table', 'data'),
    State('error-spending-table', 'columns'))
def add_row(n_clicks, n_analyses, rows, columns):
    ctx = dash.callback_context

    if not ctx.triggered:
        raise PreventUpdate

    if np.any([item["prop_id"] == 'n_analyses.value' for item in ctx.triggered]):
        old_n_analyses = int(len(rows[0]) / 2)
        if old_n_analyses > n_analyses:
            dif = int(old_n_analyses - n_analyses)
            rows = [{**{'alpha-{}'.format(i): rows[j]['alpha-{}'.format(i + dif)] for i in range(n_analyses)},
                     **{'beta-{}'.format(i): rows[j]['beta-{}'.format(i + dif)] for i in range(n_analyses)}}
                    for j in range(len(rows))]
        if old_n_analyses < n_analyses:
            dif = int(n_analyses - old_n_analyses)
            rows = \
                [{**{'alpha-{}'.format(i): 0 for i in range(dif)},
                  **{'alpha-{}'.format(i + dif): rows[j]['alpha-{}'.format(i)] for i in range(old_n_analyses)},
                  **{'beta-{}'.format(i): 0 for i in range(dif)},
                  **{'beta-{}'.format(i + dif): rows[j]['beta-{}'.format(i)] for i in range(old_n_analyses)}}
                 for j in range(len(rows))]

    if np.any([item["prop_id"] == 'adding-rows-button.n_clicks' for item in ctx.triggered]):
        rows.append({c['id']: rows[-1][c['id']] for c in columns})

    return rows
# endregion


# region critical bound simulation
def create_evaluation(local, memory_limit):
    @dash_app.callback(
        Output('status', 'children'),
        Output('status', 'color'),
        Output('status', 'is_open'),
        Output('identify_model', 'data'),
        Output('estimates', 'data'),
        Input('button', 'n_clicks'),
        State('n_analyses', 'value'),
        State('n_groups', 'value'),
        State('sample_sizes', 'derived_viewport_data'),
        State('alpha', 'value'),
        State('beta', 'value'),
        State('spending', 'value'),
        State('relative-tolerance', 'value'),
        State('CI', 'value'),
        State('costs', 'data'),
        State('error_type', 'value'),
        State('information-ratio-table', 'data'),
        State('error-spending-function', 'value'),
        State('error-spending-table', 'data'),
        State('stat_test', 'value'),
        State({'type': 'test parameter', 'name': ALL, 'form': 'value'}, 'value'),
        State({'type': 'test parameter', 'name': ALL, 'form': 'value'}, 'id'),
        State({'type': 'test parameter', 'name': ALL, 'form': 'datatable'}, 'data'),
        State({'type': 'test parameter', 'name': ALL, 'form': 'datatable'}, 'id'))
    def check_n_evaluate(n_clicks, n_analyses, n_groups, sample_sizes, alpha, beta, spending, rel_tol, CI, costs,
                         error_type, taus, error_spending_function, error_spent, stat_test, test_param_values,
                         test_param_values_ids, test_param_data, test_param_data_ids):
        """ Evaluate and simulate the properties of the design
        Step 1: Check if the user input is complete and makes sense
        Step 2: Calculate the exact properties, i.e. the properties at the first analysis.
        Step 3: Simulate the remaining properties until the desired precision level has been reached."""

        if n_clicks is None:
            raise PreventUpdate

        error_color = 'warning'

        # region check user input
        # region check small input fields
        if n_analyses is None:
            return 'Please make sure you enter a valid number of analyses.', \
                   error_color, True, dash.no_update, dash.no_update
        if n_groups is None:
            return 'Please make sure you enter a valid number of experimental groups.', \
                   error_color, True, dash.no_update, dash.no_update

        if alpha is None or beta is None:
            return 'Please make sure you fill in a probability for the type I and type II error.', \
                   error_color, True, dash.no_update, dash.no_update
        if (alpha == 1 or alpha == 0) or (beta == 1 or beta == 0):
            return 'The type I and type II errors should be between 0 and 1, they cannot be equal to 0 or 1. ' + \
                   'Please check your input', error_color, True, dash.no_update, dash.no_update

        if rel_tol is None or CI is None:
            return 'Please make sure the simulation precision parameters are filled in.', \
                   error_color, True, dash.no_update, dash.no_update
        if CI == 1 or CI == 0:
            return 'The confidence level should be between 0 and 1, but cannot be equal to 0 or 1. ' + \
                   'Please check your input', error_color, True, dash.no_update, dash.no_update
        # endregion

        problem, message = TestObject(stat_test).check_sample_size(sample_sizes, n_analyses)
        if problem:
            return message, error_color, True, dash.no_update, dash.no_update

        sample_sizes = message

        # region check costs
        if np.any([costs[0]['analysis-{}'.format(i)] == '' for i in range(n_analyses)]):
            return 'Please fill in all cells for the costs input table', \
                   error_color, True, dash.no_update, dash.no_update

        costs = np.array(pd.DataFrame(costs), dtype='d')

        if np.any(costs[1:] < costs[:-1]):
            return 'The total costs should not decrease between two analyses. Please check your input.', \
                   error_color, True, dash.no_update, dash.no_update
        if np.any(costs <= 0):
            'Experiments are not free. Please make sure all costs are larger than 0.'
        # endregion

        problem, message = check_form_error_spent(error_type, n_analyses, spending, alpha, beta,
                                                  taus, error_spending_function, error_spent)
        if problem:
            return message, error_color, True, dash.no_update, dash.no_update

        alphas = message[0]
        betas = message[1]
        error_spending_param = message[2]

        problem, message = TestObject(stat_test).check_input(test_param_values, test_param_values_ids, test_param_data,
                                                             test_param_data_ids, n_groups=n_groups)
        if problem:
            return message, error_color, True, dash.no_update, dash.no_update
        else:
            test_param_values = message
        # endregion

        # region Summarize user input, plus give id's to the different designs/models
        identify_model = {'Test': stat_test, 'Number of analyses': n_analyses, 'Sample sizes': sample_sizes.tolist(),
                          'Costs': costs.tolist(), 'Type I': alpha, 'Type II': beta, 'Spending': spending,
                          **test_param_values, **error_spending_param}
        # endregion

        # region Simulate until the confidence interval for relative error is smaller than tolerance level
        exact_sig, exact_fut, exact_true_neg, exact_power, lower_limit, upper_limit = \
            TestObject(stat_test).give_exact(sample_sizes, alphas, betas, test_param_values)

        def simulator_h0(n_sim):
            return TestObject(stat_test).simulate_statistics(n_sim, sample_sizes, 'H0', memory_limit, test_param_values)

        def simulator_ha(n_sim):
            return TestObject(stat_test).simulate_statistics(n_sim, sample_sizes, 'HA', memory_limit, test_param_values)

        # Name all the properties being simulated
        col_names = ['Model id'] + ['Sig. bound {}'.format(i + 1) for i in range(n_analyses)] + \
                    ['Fut. bound {}'.format(i + 1) for i in range(n_analyses)] + \
                    ['Expected cost H0', 'Expected cost HA'] + \
                    ['Power at analysis {}'.format(i + 1) for i in range(n_analyses)] + \
                    ['Chance of true negative under H0 at analysis {}'.format(i + 1) for i in range(n_analyses)]

        estimates, std_errors, n_simulations, counts = \
            simulation_loop(alphas, betas, exact_sig, exact_fut, rel_tol, CI, col_names, identify_model['Model id'],
                            _default_n_repeats, _max_n_repeats, simulator_h0, simulator_ha, costs, exact_true_neg,
                            exact_power, lower_limit, upper_limit)
        # endregion

        estimates = estimates.astype(str)
        std_errors = std_errors.astype(str)
        # The string type-casting is because json serialization does not support infinite values

        counts_str = '{}'.format(counts[col_names[1]][0])
        for i in range(counts.shape[0] - 1):
            counts_str += ', ' + '{}'.format(counts[col_names[1]][i])

        return [html.B('Simulations finished: '), 'Results per model based on respectively ' + counts_str +
                                                  ' estimates with {} simulations each'.format(n_simulations)], \
            'success', True, identify_model, [estimates.to_json(orient='split'), std_errors.to_json(orient='split')]


@dash_app.callback(
    Output('table', 'children'),
    Output('used_model', 'options'),
    Input('estimates', 'data'),
    State('CI', 'value'),
    State('identify_model', 'data'))
def print_the_table(df, CI, model_info):
    """ Summarize results, round to significant digits and show on webpage."""

    if df is None or CI is None:
        raise PreventUpdate

    estimates = pd.read_json(df[0], orient='split')
    estimates.index = np.arange(len(estimates))
    std_errors = pd.read_json(df[1], orient='split')
    std_errors.index = np.arange(len(std_errors))
    z_score = abs(norm.ppf(0.5 * (1 - CI)))

    sides = 1
    if 'sides' in model_info.keys() and model_info['sides'] == 'two':
        sides = 2

    n_analyses = int(model_info['Number of analyses'])

    def round_to_sig(x, x_se):
        # NaNs and infinity cannot be rounded and cannot be shown as that
        # type in a dash.DataTable. Hence -> str
        if np.isnan(x):
            return 'NaN'
        elif np.isinf(x):
            if x > 0:
                return 'Inf'
            else:
                return '- Inf'
        elif x == 0:
            return x
        elif np.isnan(x_se) or x_se == 0:
            return round(x, 9 - int(np.floor(np.log10(np.abs(x)))))
        else:
            return round(x, -int(np.floor(np.log10(z_score*x_se))))

    results_dict = [{col: round_to_sig(estimates[col][i], std_errors[col][i])
                     for col in estimates.columns[:-1]} for i in estimates.index]

    for (i, modId) in enumerate(estimates['Model id']):
        # Add model id column to the dictionary
        results_dict[i]["Model id"] = estimates["Model id"][i]

    table1 = dash_table.DataTable(columns=[{'name': 'Model id', 'id': 'Model id'},
                                           {'name': 'Power', 'id': 'Power at analysis {}'.format(n_analyses)},
                                           {'name': 'Expected cost H0', 'id': 'Expected cost H0'},
                                           {'name': 'Expected cost HA', 'id': 'Expected cost HA'}],
                                  data=results_dict, editable=False, **table_style,
                                  style_table={'overflowX': 'auto', 'maxWidth': '50rem'})

    rel_cols = []

    if model_info['Spending'] == 'alpha' or model_info['Spending'] == 'both':
        # Hide the futility bounds when only using alpha spending
        rel_cols = rel_cols + ['Sig. bound {}'.format(i + 1) for i in range(n_analyses - 1)]

    rel_cols = rel_cols + ['Sig. bound {}'.format(n_analyses)]

    if model_info['Spending'] == 'beta' or model_info['Spending'] == 'both':
        # Hide the infinite significance bounds when only using beta spending
        rel_cols = rel_cols + ['Fut. bound {}'.format(i + 1) for i in range(n_analyses)]

    table2 = dash_table.DataTable(columns=[{"name": item, "id": item} for item in ['Model id'] + rel_cols],
                                  data=results_dict, editable=False, **table_style,
                                  style_table={'overflowX': 'auto', 'maxWidth': '{}rem'.format((len(rel_cols)+2) * 10)})

    def crit_p(x, x_se, N):
        # NaNs and infinity cannot be rounded and cannot be shown as that
        # type in a dash.DataTable. Hence -> str
        if np.isnan(x):
            return 'NaN'
        elif np.isinf(x):
            if x > 0:
                return 0
            else:
                return 1
        elif np.isnan(x_se) or x_se == 0:
            p = sides * TestObject(model_info['Test']).get_p_equivalent(x, N)
            return round(p, 9 - int(np.floor(np.log10(p))))
        else:
            ps = sides * TestObject(model_info['Test']).get_p_equivalent(x, N)
            pll = sides * TestObject(model_info['Test']).get_p_equivalent(x + z_score*x_se, N)
            pul = sides * TestObject(model_info['Test']).get_p_equivalent(x - z_score*x_se, N)
            dif = max(pul-ps, ps-pll)
            return round(ps, -int(np.floor(np.log10(dif))))

    sample_sizes = np.asarray(model_info['Sample sizes'])

    results_dict = [{col: crit_p(estimates[col][i], std_errors[col][i], sample_sizes[:, int(col[-1]) - 1])
                    for col in rel_cols} for i in estimates.index]
    
    for (i, modId) in enumerate(estimates['Model id']):
        # Add model id column to the dictionary
        results_dict[i]["Model id"] = estimates["Model id"][i]

    table3 = dash_table.DataTable(columns=[{"name": item, "id": item} for item in ['Model id'] + rel_cols],
                                  data=results_dict, editable=False, **table_style,
                                  style_table={'overflowX': 'auto', 'maxWidth': '{}rem'.format((len(rel_cols)+2) * 10)})

    return [dbc.Row(dbc.Col(width={'offset': spacing['offset'], 'size': spacing['size']},
                            children=[table1, html.Br()])),
            dbc.Row(dbc.Col(width={'offset': spacing['offset'], 'size': spacing['size']},
                            children=[label('Critical values for the test statistic'), html.Br(),
                                      table2, html.Br()])),
            dbc.Row(dbc.Col(width={'offset': spacing['offset'], 'size': spacing['size']},
                            children=[label('Critical values for the p-value'), html.Br(),
                                      'Please keep in mind that after the first analysis the reported p-values no '
                                      'longer match their traditional definition.',  html.Br()])),
            dbc.Row(dbc.Col(width={'offset': spacing['offset'], 'size': spacing['size']},
                            children=[table3, html.Br()]))], \
        [{'label': str(item["Model id"]), 'value': str(item["Model id"])} for item in results_dict]


@dash_app.callback(Output('download', 'data'),
                   Input('csv_button', 'n_clicks'),
                   Input('excel_button', 'n_clicks'),
                   State('identify_model', 'data'),
                   State('estimates', 'data'))
def generate_download_file(n_clicks_csv, n_clicks_excel, identify_model, df):
    """ Return the file containing the estimates based on the simulation
    + the corresponding standard errors.
    The user can either choose to download a csv file or an excel file (different buttons).
    The csv file has just the estimates and SEs
    The excel also has a separate spreadsheet with the design input."""

    # Callback context lets us know which button was clicked
    ctx = dash.callback_context

    if not ctx.triggered:
        raise PreventUpdate

    if df is None:
        raise PreventUpdate

    # Read simulations, calculate estimates and standard errors.
    estimates = pd.read_json(df[0], orient='split')
    std_errors = pd.read_json(df[1], orient='split')

    newCols = ['Model id']
    for (i, col) in enumerate(estimates.columns[:-1]):
        newCols = newCols + [col, 'SE {}'.format(i+1)]
        estimates['SE {}'.format(i+1)] = std_errors[col]

    estimates = estimates.reindex(columns=newCols)

    if np.any([item["prop_id"] == 'csv_button.n_clicks' for item in ctx.triggered]):
        return send_data_frame(estimates.to_csv, filename="your_design.csv")

    # Create the estimates and confidence intervals for the p-values
    sides = 1
    if 'sides' in identify_model.keys() and identify_model['sides'] == 'two':
        sides = 2

    n_analyses = int(identify_model['Number of analyses'])

    p_cols = ['Sig. bound {}'.format(i + 1) for i in range(n_analyses)] + \
             ['Fut. bound {}'.format(i + 1) for i in range(n_analyses)]

    def crit_p(x, x_se, N, col_name):
        # NaNs and infinity cannot be rounded and cannot be shown as that
        # type in a dash.DataTable. Hence -> str
        if np.isnan(x):
            return {col_name: 'NaN', '95%CI (lower): ' + col_name: 'NaN', '95%CI (upper): ' + col_name: 'NaN'}
        elif np.isinf(x):
            if x > 0:
                return {col_name: 0, '95%CI (lower): ' + col_name: 'NaN', '95%CI (upper): ' + col_name: 'NaN'}
            else:
                return {col_name: 1, '95%CI (lower): ' + col_name: 'NaN', '95%CI (upper): ' + col_name: 'NaN'}
        else:
            return {col_name: sides * TestObject(identify_model['Test']).get_p_equivalent(x, N),
                    '95%CI (lower):' + col_name:
                        sides * TestObject(identify_model['Test']).get_p_equivalent(x+norm.ppf(0.975) * x_se, N),
                    '95%CI (upper):' + col_name:
                        sides * TestObject(identify_model['Test']).get_p_equivalent(x-norm.ppf(0.975) * x_se, N)}

    sample_sizes = np.asarray(identify_model['Sample sizes'])

    p_dicts = [{'Model id': identify_model['Model id'][i]} for i in estimates.index]

    [p_dict.update(crit_p(estimates[col][i], std_errors[col][i], sample_sizes[:, int(col[-1]) - 1], col))
     for col in p_cols for (i, p_dict) in enumerate(p_dicts)]

    # Create excel file if button clicked

    if np.any([item["prop_id"] == 'excel_button.n_clicks' for item in ctx.triggered]):
        model_specs = pd.DataFrame()
        for key in identify_model:
            ar = np.asarray(identify_model[key])
            if len(ar.shape) == 2:
                if ar.shape[1] > 1:
                    colNames = ['{} at analysis {}'.format(key, i + 1) for i in range(ar.shape[1])]
                    model_specs = pd.concat([model_specs, pd.DataFrame(ar, columns=colNames)], axis=1)
                else:
                    model_specs = pd.concat([model_specs, pd.DataFrame(ar, columns=[key])], axis=1)
            else:
                model_specs = pd.concat([model_specs, pd.DataFrame(ar.reshape((ar.size, 1)), columns=[key])], axis=1)

        def to_xlsx(bytes_io):
            writer = pd.ExcelWriter(bytes_io, engine="xlsxwriter")
            model_specs.to_excel(writer, index=False, sheet_name='Design input')
            estimates.to_excel(writer, index=False, sheet_name='Estimates')
            pd.DataFrame(p_dicts).to_excel(writer, index=False, sheet_name='P-value estimates')
            writer.save()

        return send_bytes(to_xlsx, "your_design.xlsx")

# endregion


@dash_app.callback(Output('n_termination', 'min'),
                   Output('n_termination', 'max'),
                   Input('used_model', 'value'),
                   State('identify_model', 'data'))
def update_analysis_number(model_id, model_info):
    if model_id is None or model_info is None:
        raise PreventUpdate
    return 1, model_info["Number of analyses"]


def create_evaluation_CI(local, memory_limit):
    @dash_app.callback(Output('effect_estimate', 'children'),
                       Input('ci_button', 'n_clicks'),
                       State('used_model', 'value'),
                       State('identify_model', 'data'),
                       State('estimates', 'data'),
                       State('n_termination', 'value'),
                       State('result_statistic', 'value'),
                       State('ci_confidence', 'value'),
                       State('rel-tol-CI', 'value'),
                       State('max_iter', 'value'))
    def simulate_CI(n_clicks, model_id, model_info, df, n_termination, T, confidence_level, rel_tol, max_iter):
        if n_clicks is None:
            raise PreventUpdate

        # region check user input and transform
        if model_id is None:
            return "Please select the used design."

        n_analyses = model_info['Number of analyses']
        sample_sizes = np.asarray(model_info['Sample sizes'], dtype=int)
        test = model_info['Test']
        if test == 'One-way':
            one_sided = True
        elif test == 'T':
            one_sided = model_info['sides'] == 'one'
        else:
            raise ValueError

        estimates = pd.read_json(df[0], orient='split')
        model_n = estimates['Model id'] == model_id
        sig_bounds = np.zeros(n_analyses)
        fut_bounds = np.zeros(n_analyses)

        for i in range(n_analyses):
            sig_bounds[i] = estimates['Sig. bound {}'.format(i + 1)][model_n]
            fut_bounds[i] = estimates['Fut. bound {}'.format(i + 1)][model_n]

        if fut_bounds[n_termination - 1] < T < sig_bounds[n_termination - 1]:
            return 'The test statistic should be either larger than the significance bound or smaller than the ' \
                   'futility bound at the termination of the experiment. Please check your input.'

        if confidence_level == 0 or confidence_level == 1:
            'The confidence level should be strictly larger than zero and strictly smaller than one.'

        alpha = 1 - confidence_level

        # endregion

        relevant_info = fut_bounds

        estimate_text = simulate_effect_CI(n_termination, T, sig_bounds, fut_bounds, sample_sizes, alpha, test,
                                           one_sided, rel_tol, rel_tol, memory_limit, 10**max_iter)

        return estimate_text
