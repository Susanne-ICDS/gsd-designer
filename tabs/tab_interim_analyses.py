from dash import dash_table, html, dcc
import dash_bootstrap_components as dbc

from statistical_parts.statistical_test_objects import test_options, TestObject

from layout_instructions import spacing_variables as spacing
from layout_instructions import label, my_jumbo_box, table_style

layout = html.Div([
    my_jumbo_box('Interim analyses', 'Changes over time'),

    html.Br(),
    dbc.Row(dbc.Col(dbc.Alert(id='fixed-n', is_open=False, dismissable=True),
                    width={'offset': spacing['offset'], 'size': 'auto'})),

    html.Div(id='test_input_tab2',
             children=TestObject(test_options[0]["value"]).tab_interim_analyses()),

    html.Br(),
    dbc.Row(dbc.Col(width={'offset': spacing['offset'], 'size': spacing['size']},
                    children=[
                        label('Total costs at analysis'),
                        dbc.Checklist(options=[{'label': 'Default option: the costs equal the sample sizes',
                                                'value': 'default'}],
                                      value=['default'],
                                      id='cost-default',
                                      switch=True)])),

    dbc.Row(dbc.Col(width={'offset': spacing['offset'], 'size': 'auto'},
                    children=[dash_table.DataTable(id='costs', columns=[], data=[], editable=True, **table_style)]))
])
