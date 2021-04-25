import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html

import git

from app import app, server
import callbacks

from tabs import navbar, tab_basic_design, tab_interim_analyses, tab_error_spending, tab_simulation

from layout_instructions import spacing_variables as spacing

# Get the most recent commit date
repo = git.Repo("./.git")
headcommit = repo.head.commit
headcommit.committed_date

app.layout = html.Div([
    navbar.bar,
    dcc.Location(id='url', pathname='/basic-design', refresh=False),

    html.Div(id='basic-design', hidden=False, children=tab_basic_design.layout),
    html.Div(id='interim-analyses', hidden=True, children=tab_interim_analyses.layout),
    html.Div(id='error-spending', hidden=True, children=tab_error_spending.layout),
    html.Div(id='simulation', hidden=True, children=tab_simulation.layout),

    html.Br(),
    html.Br(),

    dbc.Row([dbc.Col(width={'size': 'auto', 'offset': spacing['offset']},
                     children=dbc.ButtonGroup([dbc.Button('<< Back', id='previous', color='primary'),
                                               dbc.Button('Next step >>', id='next', color='primary')])),
             dbc.Col(width=spacing['offset'])],
            justify='end'),

    html.Br(),
    dbc.Row([dbc.Col(width={'size': 'auto', 'offset': spacing['offset']},
                     children=['BISI logo and info', html.Br(), 'ICDS logo and info', html.Br(),
                               ])],
            dbc.Col(width={'size': 'auto', 'order': 12},
                    children=['App developed by: Susanne Blotwijk', html.Br(),
                              '{}'.format(1)]),
            style={'color': 'white', 'background-color': 'dark'}),


    dcc.Store(id='identify_model', storage_type='session'),
    # Summary of the user input

    dcc.Store(id='estimates', storage_type='session'),
    # The simulations of the critical values, power etc.
    # This is a list of two json serialized pandas dataframes 1) the estimates and 2) their standard errors.

])

if __name__ == '__main__':
    app.run_server(debug=True)
