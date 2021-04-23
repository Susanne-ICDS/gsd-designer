import dash_html_components as html
import dash_core_components as dcc
import dash_table

from test_submodule import error_spending

layout = html.Div([
    html.H4('Error spending'),
    html.P('How do you wish to distribute your probability of type I and type II errors?'),
    dcc.RadioItems(id='error_type', options=[{'label': 'Use spending functions', 'value': 'IR'},
                                             {'label': 'Enter allowed errors directly', 'value': 'DES'}], value='IR'),

    html.Label('Which forms of error spending would you like to perform?'),
    dcc.Dropdown(id='spending', options=[{'label': 'alpha and beta spending', 'value': 'both'},
                                         {'label': 'just alpha spending', 'value': 'alpha'},
                                         {'label': 'just beta spending', 'value': 'beta'}], value='both'),

    error_spending.layout,
])
