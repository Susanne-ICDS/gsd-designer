import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html

import git
import time

from app import app, server
import callbacks

from tabs import navbar, tab_basic_design, tab_interim_analyses, tab_error_spending, tab_simulation

from layout_instructions import spacing_variables as spacing


# Get the most recent commit date
# repo = git.Repo("./.git")
# head_commit = repo.head.commit
# time.asctime(time.gmtime(head_commit.committed_date))

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

    dbc.Row([dbc.Col(width={'size': 3, 'offset': spacing['offset']},
                     children=['BISI logo and info',
                               html.Br(), 'ICDS logo and info',
                               html.Br(), 'Laarbeeklaan etc. etc.',
                               html.Br(), 'mail?'
                               ]),
             dbc.Col(width={'size': 3},
                     children=['Code and documentation on ',
                               html.A('Github', href='https://github.com/Susanne-ICDS/gsd-designer',
                                      style={'color': '#fff'}),
                               html.Br(), 'Publication: Coming soon',
                               html.Br(), 'Document with example: Coming soon'
                               ]),
             dbc.Col(width={'size': 3},
                     children=['App developed by: Susanne Blotwijk',
                               html.Br(), 'Most recent update: 25 Apr. 2021'
                               # '{}'.format(time.strftime("%d %b %Y, %H:%M", time.gmtime(head_commit.committed_date)))
                               ])],
            style={'color': '#fff', 'background-color': '#1E1E1E', 'font-family': 'roboto'},
            justify="center"),

    dbc.Row(style={'color': '#fff', 'background-color': '#1E1E1E', "height": "2rem"}),

    dcc.Store(id='identify_model', storage_type='session'),
    # Summary of the user input

    dcc.Store(id='estimates', storage_type='session'),
    # The simulations of the critical values, power etc.
    # This is a list of two json serialized pandas dataframes 1) the estimates and 2) their standard errors.

], fluid=True)])

if __name__ == '__main__':
    app.run_server(debug=True)
