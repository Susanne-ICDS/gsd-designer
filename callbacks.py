import dash
import dash_table
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

_default_n_repeats = 100
_max_n_repeats = 10 ** 6


@app.callback(
    Output('basic-design', 'hidden'),
    Output('interim-analyses', 'hidden'),
    Output('error-spending', 'hidden'),
    Output('simulation', 'hidden'),
    Output('tab1', 'active'),
    Output('tab2', 'active'),
    Output('tab4', 'active'),
    Output('tab3', 'active'),
    Input('url', 'pathname'))
def navigation(path):
    if path is None:
        return False, True, True, True, True, False, False, False
    else:
        return path != '/basic-design', path != '/interim-analyses', path != '/error-spending', path != '/simulation', \
               path == '/basic-design', path == '/interim-analyses', path == '/error-spending', path == '/simulation'


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


# region test input
@app.callback(
    Output('costs', 'columns'),
    Output('costs', 'data'),
    Output('costs', 'style_data'),
    Output('costs', 'style_header'),
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
        return cols, dash.no_update, dash.no_update, dash.no_update

    if checked == ['default']:
        return cols, TestObject(stat_test).default_costs(n_analyses, n_groups, sample_sizes), \
               disabled_style_data, disabled_style_header
    else:
        return cols, dash.no_update, table_style['style_data'], table_style['style_header']


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
def update_test_input(n_groups, stat_test, rows):
    """ Adjust the test specific parameter input if necessary """

    if rows is not None and n_groups is not None:
        return TestObject(stat_test).update_test_parameter_input(n_groups=n_groups, rows=rows)

    raise PreventUpdate
# endregion
