import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_table

from test_submodule.statistical_test_objects import test_options, TestObject

from layout_instructions import spacing_variables as spacing
from layout_instructions import label, page_header, regular_text, vub_blue

layout = html.Div([
    dbc.Row(dbc.Col(width={'offset': spacing['offset'], 'size': spacing['size']},
                    children=[
                        html.Br(),
                        page_header('Basic design'),
                        regular_text('Design your experiment.'),
                        html.Br(),
                        regular_text(' ')]),
            style={'background-color': vub_blue}),
    html.Br(),
    dbc.Row(dbc.Col(width={'offset': spacing['offset'], 'size': spacing['size']},
                    children=[
                        label('Statistical test'),
                        dbc.Select(id="stat_test",
                                   options=[item for item in test_options],
                                   value=test_options[0]["value"])
                   ])),
    html.Br(),

    dbc.Row(dbc.Col(label('Total allowed type I and type II errors'),
                    width={'offset': spacing['offset'], 'size': spacing['size']})),

    dbc.Row([dbc.Col(dbc.Input(id='alpha', placeholder='Type I error', type='number', value=0.05, debounce=True,
                               min=0, max=1, step=10**-5), width={'offset': spacing['offset'],
                                                                  'size': spacing['float_input']}),
             dbc.Col(dbc.Input(id='beta', placeholder='Type II error', type='number', value=0.2, debounce=True,
                               min=0, max=1, step=10**-5), width=spacing['float_input'])
             ]),

    html.Div(id='test_input_tab1', children=TestObject(test_options[0]["value"]).tab_basic_design())
    ])
