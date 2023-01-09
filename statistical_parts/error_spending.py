from dash import dash_table
from dash import html
import dash_bootstrap_components as dbc

import numpy as np
import pandas as pd
from scipy.stats import norm

from layout_instructions import spacing_variables as spacing
from layout_instructions import label, my_jumbo_box, table_style

# Add a label for a new error spending function here, and add the function itself into the code of e_s_f_2_error_spent

error_spending_function_dict = [{'label': 'Pocock', 'value': 'Pocock'},
                                {'label': 'O´Brien-Fleming', 'value': 'OBF'},
                                {'label': 'Linear', 'value': 'Linear'}]

layout = html.Div(
    [html.Div(id='IR', hidden=False, children=[
        dbc.Row(dbc.Col(width={'offset': spacing['offset'], 'size': spacing['size']},
                        children=[label('Information ratio'),
                                  dbc.Checklist(id='ir-default', value=['default'], switch=True,
                                                options=[{'label': 'Use the default option: the sample size'
                                                                   ' divided by the sample size at the final analysis',
                                                          'value': 'default'}])])),
        dbc.Row(dbc.Col(width={'offset': spacing['offset'], 'size': 'auto'},
                        children=dash_table.DataTable(id='information-ratio-table', columns=[], data=[],
                                                      editable=False, **table_style))),
        html.Br(),
        dbc.Row(dbc.Col(width={'offset': spacing['offset'], 'size': spacing['size']},
                        children=[label('Error spending functions'),
                        dbc.Checklist(id='error-spending-function', options=error_spending_function_dict,
                                      value=[item['value'] for item in error_spending_function_dict])]))]
              ),

     html.Div(id='DES', hidden=True, children=[
         dbc.Row(dbc.Col(width={'offset': spacing['offset'], 'size': spacing['size']},
                         children=[label('Error spending values per analysis')])),
         dbc.Row(dbc.Col(width={'offset': spacing['offset'], 'size': 'auto'},
                 children=[dash_table.DataTable(id='error-spending-table', editable=True, row_deletable=True,
                                                columns=[{'name': 'Alpha 1', 'id': 'alpha-0'},
                                                         {'name': 'Alpha 2', 'id': 'alpha-1'},
                                                         {'name': 'Beta 1', 'id': 'beta-0'},
                                                         {'name': 'Beta 2', 'id': 'beta-1'}],
                                                data=[{'alpha-0': 0.01, 'alpha-1': 0.05,
                                                       'beta-0': 0.04, 'beta-1': 0.2}], **table_style),
                           dbc.Button('Add Row', id='adding-rows-button', n_clicks=0, color='primary', outline=True)])),
               ])])


def e_s_f_2_error_spent(alpha, taus, error_spending_function, n_analyses):
    """ Calculate alpha spent based on spending functions and information ratios (taus).

    This was written with the terminology for alpha spending, but works for beta spending as well"""

    n_models = len(error_spending_function)

    alphas = np.ones((n_models, n_analyses))
    for i in range(n_models):
        if error_spending_function[i] == 'Pocock':
            alphas[i] = alpha * np.log(1 + (np.exp(1) - 1) * taus)
        elif error_spending_function[i] == 'OBF':
            alphas[i] = 2 - 2 * norm.cdf(norm.ppf(1 - alpha / 2) / (taus ** 0.5))
        elif error_spending_function[i] == 'Linear':
            alphas[i] = alpha*taus
        else:
            return print('Spending function not implemented')
    return alphas


def check_form_error_spent(error_type, n_analyses, spending, alpha, beta, taus, error_spending_function, error_spent):
    """ Check if user input makes sense and if so return corresponding alphas and betas spent per analysis """
    n_models = 1
    if error_type == 'IR':
        n_models = len(error_spending_function)
    elif error_type == 'DES':
        n_models = len(error_spent)

    alphas = np.zeros((n_models, n_analyses))
    alphas[:, -1] = alpha
    betas = np.zeros((n_models, n_analyses))
    betas[:, -1] = beta

    if error_type == 'IR':  # information ratio type
        # region check input
        if np.any([taus[0]['analysis-{}'.format(i)] == '' for i in range(n_analyses)]):
            return True, 'Please fill in all cells for the costs input table',

        taus = np.array(pd.DataFrame(taus), dtype='d')
        taus = taus.reshape(taus.size)

        if taus[-1] != 1:
            return True, 'The information ratio of the last analysis should be 1. Please check your input'

        elif np.any(taus <= 0) or np.any(taus > 1):
            return True, 'The information ratio should be strictly larger than 0 and smaller than or equal to 1. ' + \
                   'Please check your input'
        elif np.any(taus[1:] < taus[:-1]):
            return True, 'The information ratio cannot decrease between two analyses. Please check your input'
        elif not error_spending_function:
            return True, 'Please select at least one error spending function'
        # endregion

        if spending == 'alpha' or spending == 'both':
            alphas = e_s_f_2_error_spent(alpha, taus, error_spending_function, n_analyses)

        if spending == 'beta' or spending == 'both':
            betas = e_s_f_2_error_spent(beta, taus, error_spending_function, n_analyses)

        labels = []
        for value in error_spending_function:
            for item in error_spending_function_dict:
                if value == item['value']:
                    labels = labels + [item['label']]
                    break

        return False, [alphas, betas, {'Information ratio': taus.tolist(), 'Spending functions': labels,
                                       'Model id': error_spending_function}]

    if error_type == 'DES':
        error_spent_df = pd.DataFrame(error_spent)
        if len(error_spent_df) == 0:
            return True, 'Please add at least one row of error spending values. '

        alpha_col = ['alpha-{}'.format(i) for i in range(n_analyses)]
        beta_col = ['beta-{}'.format(i) for i in range(n_analyses)]

        if spending == 'alpha' or spending == 'both':
            try:
                alphas = np.asarray(error_spent_df[alpha_col])
            except KeyError:
                return True, 'Please fill in all the cells in the error spending input table.'

        if spending == 'beta' or spending == 'both':
            try:
                betas = np.asarray(error_spent_df[beta_col])
            except KeyError:
                return True, 'Please fill in all the cells in the error spending input table.'

        if np.any(np.isnan(alphas)) or np.any(np.isnan(betas)):
            return True, 'Please fill in all the cells in the error spending input table.'

        if np.any(np.abs(alphas[:, -1] - alpha) > 10**-8):
            return True, 'The error spending values in column Alpha {} should equal the '.format(n_analyses) + \
                   'type I error, i.e. {}. Please check your input'.format(alpha),

        if np.any(np.abs(betas[:, -1] - beta) > 10**-8):
            return True, 'The error spending values in column Beta {} should equal the '.format(n_analyses) + \
                   'type II error, i.e. {}. Please check your input'.format(beta),

        if np.any(alphas[:, 1:] < alphas[:, :-1]) or np.any(betas[:, 1:] < betas[:, :-1]):
            return True, 'The error spending values cannot decrease between analyses. Please check your input'

        if np.any(alphas < 0) or np.any(betas < 0):
            return True, 'The error spending values cannot be negative.'

        return False, [alphas, betas, {'Error spending values': error_spent,
                                       'Model id': np.arange(1, len(error_spent) + 1)}]
