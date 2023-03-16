import dash_bootstrap_components as dbc
from dash import dash_table, dcc, html

from statistical_parts import error_spending

from layout_instructions import spacing_variables as spacing
from layout_instructions import label, my_jumbo_box, table_style

layout = html.Div([
    my_jumbo_box('Error spending', 'Distribute the allowed errors over the analyses'),

    html.Br(),
    dbc.Row(dbc.Col(width={'offset': spacing['offset'], 'size': spacing['size']},
                    children=[label('When would you like to stop early?'),
                              dcc.Dropdown(id='spending', options=[
                                  {'label': 'Either if the result is significant or if it is insufficiently promising',
                                   'value': 'both'},
                                  {'label': 'Only if the result is significant', 'value': 'alpha'},
                                  {'label': 'Only if the result is insufficiently promising', 'value': 'beta'}], value='both'),
                              html.Br(),
                              label('How would you like to enter the error spending?'),
                              dbc.RadioItems(id='error_type',
                                             options=[{'label': 'Use default spending functions', 'value': 'IR'},
                                                      {'label': 'Enter allowed errors directly', 'value': 'DES'}],
                                             value='IR'),
                              html.Br()
                              ])),
    error_spending.layout
])
