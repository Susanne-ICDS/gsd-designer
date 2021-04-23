import dash
import dash_table
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State, MATCH, ALL
from dash_extensions.snippets import send_data_frame, send_bytes

import pandas as pd
import numpy as np
import xlsxwriter

from app import app
from test_submodule.statistical_test_objects import TestObject
from test_submodule.error_spending import check_form_error_spent
from test_submodule.math_parts.error_spending_simulation import simulation_loop

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


# region test input
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
