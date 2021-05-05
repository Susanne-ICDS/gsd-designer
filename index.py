import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html

import datetime
import numpy as np

from app import app, server
import callbacks

from tabs import navbar, tab_basic_design, tab_interim_analyses, tab_error_spending, tab_simulation

from layout_instructions import spacing_variables as spacing

BISI_LOGO = '/assets/Logos.svg'
version = '1.1.' + str(int(open("version.txt", "r").read()))

app.layout = html.Div([navbar.bar, dbc.Container([

    dcc.Location(id='url', pathname='/basic-design', refresh=False),

    html.Div(id='basic-design', hidden=False, children=tab_basic_design.layout),
    html.Div(id='interim-analyses', hidden=True, children=tab_interim_analyses.layout),
    html.Div(id='error-spending', hidden=True, children=tab_error_spending.layout),
    html.Div(id='simulation', hidden=True, children=tab_simulation.layout),

    html.Br(),
    html.Br(),


    dbc.Row([dbc.Col(width={'size': 'auto', 'offset': spacing['offset']},
                     children=dbc.ButtonGroup([dbc.Button('<< Back', id='previous', color='primary', disabled=True),
                                               dbc.Button('Next step >>', id='next', color='primary')])),
             dbc.Col(width=spacing['offset'])],
            justify='end'),

    html.Br(),
    html.Br(),

    dbc.Row(style={'color': '#fff', 'background-color': '#1E1E1E', "height": "2rem"}),

    dbc.Row([dbc.Col(width={'size': 5},
                     children=[dbc.Container(html.A([html.Img(src=BISI_LOGO, alt='BISI Logo', height='175rem')],
                                                    href='https://bisi.research.vub.be'), fluid=True),
                               html.Br(),
                               html.Br(),
                               'Laarbeeklaan 103, 1090 Jette',
                               html.Br(), 'Brussels, Belgium',
                               html.Br(), html.Br(),
                               html.A('bisi.research.vub.be', href='https://bisi.research.vub.be',
                                      style={'color': '#fff'}),
                               html.Br(), html.A('icds.be', href='https://icds.be', style={'color': '#fff'})
                               ],
                     style={'textAlign': 'center'}),
             dbc.Col(width={'size': 3},
                     children=[html.Br(),
                               'Code and documentation: ',
                               html.A('Github', href='https://github.com/Susanne-ICDS/gsd-designer',
                                      style={'color': '#fff', 'text-decoration': 'underline'}),
                               html.Br(), html.Br(),
                               'Citable publication: Coming soon',
                               html.Br(), html.Br(),
                               'Tutorial: Coming soon'
                               ]),
             dbc.Col(width={'size': 3},
                     children=[html.Br(),
                               'App developed by: Susanne Blotwijk', html.Br(),
                               html.Br(), 'Version: ' + version, html.Br(),
                               html.Br(), 'Copyright: 2021-{}'.format(datetime.datetime.now().year),
                               ])],
            style={'color': '#fff', 'background-color': '#1E1E1E', 'font-family': 'Roboto'},
            justify="center"),

    dbc.Row(style={'color': '#fff', 'background-color': '#1E1E1E', "height": "2rem"}),

    dcc.Store(id='identify_model', storage_type='session'),
    # Summary of the user input

    dcc.Store(id='estimates', storage_type='session'),
    # The simulations of the critical values, power etc.
    # This is a list of two json serialized pandas dataframes 1) the estimates and 2) their standard errors.

], fluid=True)])

local = False
memory_limit = 1 - 0.25  # in GigaByte
callbacks.create_evaluation(local, memory_limit)

if __name__ == '__main__':
    app.run_server(debug=True)
