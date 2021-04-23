import dash_html_components as html
import dash_core_components as dcc
import dash_table

layout = html.Div([
    html.H4('Interim analyses'),
    html.P('How many analyses do you wish to perform and what changes between them?'),

    html.Div(id='test_input_tab2',
             children=[html.Div([dcc.Input(id='n_analyses', value=2), dash_table.DataTable(id='sample_sizes')],
                                hidden=True)]),

    html.Label('Total costs at analysis'),
    dcc.Checklist(id='cost-default', options=[{'label': 'Default option: costs = sample sizes', 'value': 'default'}],
                  value=['default']),
    dash_table.DataTable(id='costs', columns=[], data=[], editable=True)
])
