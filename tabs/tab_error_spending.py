import dash_html_components as html
import dash_bootstrap_components as dbc
from dash import dash_table, dcc

from statistical_parts import error_spending

from layout_instructions import spacing_variables as spacing
from layout_instructions import label, my_jumbo_box, table_style

layout = html.Div([
    my_jumbo_box('Error spending', 'Distribute the allowed errors over the analyses'),

    html.Br(),
    dbc.Row(dbc.Col(width={'offset': spacing['offset'], 'size': spacing['size']},
                    children=[label('Which forms of error spending would you like to perform?'),
                              dcc.Dropdown(id='spending',
                                           options=[{'label': 'alpha and beta spending', 'value': 'both'},
                                                    {'label': 'just alpha spending', 'value': 'alpha'},
                                                    {'label': 'just beta spending', 'value': 'beta'}], value='both'),
                              html.Br(),
                              label('How would you like to enter the error spending?'),
                              dbc.RadioItems(id='error_type',
                                             options=[{'label': 'Use spending functions', 'value': 'IR'},
                                                      {'label': 'Enter allowed errors directly', 'value': 'DES'}],
                                             value='IR'),
                              html.Br()
                              ])),
    error_spending.layout
])
