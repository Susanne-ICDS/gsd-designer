import dash
import dash_table
import dash_html_components as html
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State, MATCH, ALL
from dash_extensions.snippets import send_data_frame, send_bytes

import pandas as pd
import numpy as np
import xlsxwriter

from app import app

from statistical_parts.statistical_test_objects import TestObject
from statistical_parts.error_spending import check_form_error_spent
from statistical_parts.math_parts.error_spending_simulation import simulation_loop

from layout_instructions import table_style, disabled_style_header, disabled_style_data

_default_n_repeats = 10
_max_n_repeats = 10 ** 6


@app.callback(
    Output('basic-design', 'hidden'),
    Output('interim-analyses', 'hidden'),
    Output('error-spending', 'hidden'),
    Output('simulation', 'hidden'),
    Output('tab1', 'active'),
    Output('tab2', 'active'),
    Output('tab3', 'active'),
    Output('tab4', 'active'),
    Output('previous', 'disabled'),
    Output('next', 'disabled'),
    Output('previous', 'href'),
    Output('next', 'href'),
    Input('url', 'pathname'))
def navigation(path):
    if path is None:
        return False, True, True, True, True, False, False, False, True, False, None, '/interim-analyses'

    first = path == '/basic-design'
    last = path == '/simulation'
    tabs = np.array(['/basic-design', '/interim-analyses', '/error-spending', '/simulation'])
    current_tab = int(np.arange(0, 4)[tabs == path])
    previous_tab = None
    next_tab = None

    if current_tab > 0:
        previous_tab = tabs[current_tab - 1]
    if current_tab < 3:
        next_tab = tabs[current_tab + 1]

    return path != '/basic-design', path != '/interim-analyses', path != '/error-spending', path != '/simulation', \
        path == '/basic-design', path == '/interim-analyses', path == '/error-spending', path == '/simulation', \
        first, last, previous_tab, next_tab


@app.callback(
    Output('test_input_tab1', 'children'),
    Input('stat_test', 'value'))
def display_tab1(stat_test):
    return TestObject(stat_test).tab_basic_design()


@app.callback(
    Output('test_input_tab2', 'children'),
    Input('stat_test', 'value'))
def display_tab2(stat_test):
    return TestObject(stat_test).tab_interim_analyses()


@app.callback(Output('explain-accuracy', 'children'),
              Input('relative-tolerance', 'value'),
              Input('CI', 'value'))
def explain_accuracy(rel_tol, CI):
    if rel_tol is None or CI is None:
        raise PreventUpdate

    return 'Simulations will continue until the radius of the {}%-confidence '.format(round(CI*100, ndigits=2)) + \
           'interval has a radius of less than {}% of the estimate. '.format(round(rel_tol*100, ndigits=2)) + \
           'The relative tolerance level determines to how many significant figures the results are rounded.'


# region test input
@app.callback(
    Output('costs', 'style_data'),
    Output('costs', 'style_header'),
    Output('costs', 'editable'),
    Input('cost-default', 'value'))
def disable_costs(checked):
    if checked == ['default']:
        return disabled_style_data, disabled_style_header, False
    else:
        return table_style['style_data'], table_style['style_header'], True


@app.callback(
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


@app.callback(
    Output('sample_sizes', 'columns'),
    Input('n_analyses', 'value'))
def resize_columns(n_analyses):
    """ Adjust the size of the sample size input table to the number of analyses """
    if n_analyses is None:
        raise PreventUpdate

    return [{'name': 'Analysis {}'.format(i + 1), 'id': 'analysis-{}'.format(i), 'type': 'numeric'}
            for i in range(n_analyses)]


@app.callback(
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


@app.callback(
    Output({'type': 'test parameter', 'name': MATCH, 'form': 'datatable'}, 'data'),
    Input('n_groups', 'value'),
    State('stat_test', 'value'),
    State({'type': 'test parameter', 'name': MATCH, 'form': 'datatable'}, 'data'))
def update_input_table(n_groups, stat_test, rows):
    """ Adjust the test specific parameter input if necessary """

    if rows is not None and n_groups is not None:
        return TestObject(stat_test).update_input_table(n_groups=n_groups, rows=rows)

    raise PreventUpdate


@app.callback(
    Output('fixed-n', 'is_open'),
    Output('fixed-n', 'color'),
    Output('fixed-n', 'children'),
    Input('url', 'pathname'),
    State('stat_test', 'value'),
    State('alpha', 'value'),
    State('beta', 'value'),
    State({'type': 'test parameter', 'name': ALL, 'form': 'value'}, 'value'),
    State({'type': 'test parameter', 'name': ALL, 'form': 'value'}, 'id'),
    State({'type': 'test parameter', 'name': ALL, 'form': 'datatable'}, 'data'),
    State({'type': 'test parameter', 'name': ALL, 'form': 'datatable'}, 'id')
)
def give_n(path, stat_test, alpha, beta, test_param_values, test_param_values_ids, test_param_data,
           test_param_data_ids):
    if path == '/interim-analyses':
        color, message = TestObject(stat_test).fixed_sample_size(alpha, beta, test_param_values, test_param_values_ids,
                                                                 test_param_data, test_param_data_ids)
        return True, color, message
    raise PreventUpdate
# endregion


# region error-spending input
@app.callback(
    Output('IR', 'hidden'),
    Output('DES', 'hidden'),
    Input('error_type', 'value'))
def switch_type(error_type):
    """ Display the chosen input type, hide the other """

    return 'IR' != error_type, 'DES' != error_type


@app.callback(
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


@app.callback(
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


@app.callback(
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


@app.callback(
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


def create_evaluation(local, memory_limit):
    @app.callback(
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
        exact_sig, exact_fut, exact_true_neg, exact_power = \
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
                            exact_power)
        # endregion

        estimates = estimates.astype(str)
        std_errors = std_errors.astype(str)
        # The string type-casting is because json serialization does not support infinite values

        return [html.B('Simulations finished: '), 'Results based on {} estimates'.format(np.asarray(counts[col_names[1]])) +
                                                  ' with {} simulations each'.format(n_simulations)], \
            'success', True, identify_model, [estimates.to_json(orient='split'), std_errors.to_json(orient='split')]


@app.callback(
    Output('table', 'children'),
    Input('estimates', 'data'),
    State('relative-tolerance', 'value'),
    State('identify_model', 'data'))
def print_the_table(df, relTol, model_info):
    """ Summarize results, round to significant digits and show on webpage."""

    if df is None:
        raise PreventUpdate

    estimates = pd.read_json(df[0], orient='split')

    def round_to_sig(x):
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
        else:
            return round(x, int(-np.floor(np.log10(abs(relTol))) - (np.floor(np.log10(abs(x))))))

    results_dict = [{col: round_to_sig(estimates[col][i]) for col in estimates.columns[:-1]} for i in estimates.index]

    for (i, modId) in enumerate(estimates['Model id']):
        # Add model id column to the dictionary
        results_dict[i]["Model id"] = estimates["Model id"][i]

    n_analyses = int(model_info['Number of analyses'])

    # Only show model model id, the relevant critical values, total power and projected costs
    column_order = ['Model id']

    if model_info['Spending'] == 'alpha' or model_info['Spending'] == 'both':
        # Hide the futility bounds when only using alpha spending
        column_order = column_order + ['Sig. bound {}'.format(i + 1) for i in range(n_analyses - 1)]

    column_order = column_order + ['Sig. bound {}'.format(n_analyses)]

    if model_info['Spending'] == 'beta' or model_info['Spending'] == 'both':
        # Hide the infinite significance bounds when only using beta spending
        column_order = column_order + ['Fut. bound {}'.format(i + 1) for i in range(n_analyses)]

    column_order = column_order + ['Power at analysis {}'.format(n_analyses), 'Expected cost H0', 'Expected cost HA']
    column_names = column_order[:-3] + ['Power'] + column_order[-2:]

    return dash_table.DataTable(columns=[{"name": column_names[i], "id": item} for (i, item) in enumerate(column_order)],
                                data=results_dict, editable=False, **table_style)


@app.callback(Output('download', 'data'),
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

    # Read simulations, calculate estimates and standard errors.
    estimates = pd.read_json(df[0], orient='split')
    std_errors = pd.read_json(df[1], orient='split')

    newCols = ['Model id']
    for (i, col) in enumerate(estimates.columns[1:]):
        newCols = newCols + [col, 'SE {}'.format(i+1)]
        estimates['SE {}'.format(i+1)] = std_errors[col]

    estimates = estimates.reindex(columns=newCols)

    if np.any([item["prop_id"] == 'csv_button.n_clicks' for item in ctx.triggered]):
        return send_data_frame(estimates.to_csv, filename="your_design.csv")

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
            writer.save()

        return send_bytes(to_xlsx, "your_design.xlsx")
