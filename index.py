import dash_core_components as dcc
import dash_html_components as html

from app import app, server
import callbacks
from tabs import navbar, tab_basic_design, tab_interim_analyses, tab_error_spending, tab_simulation

app.layout = html.Div([
    navbar.bar,
    dcc.Location(id='url', pathname='/basic-design', refresh=False),

    html.Div(id='basic-design', hidden=False, children=tab_basic_design.layout),
    html.Div(id='interim-analyses', hidden=True, children=tab_interim_analyses.layout),
    html.Div(id='error-spending', hidden=True, children=tab_error_spending.layout),
    html.Div(id='simulation', hidden=True, children=tab_simulation.layout),

    dcc.Store(id='identify_model', storage_type='session'),
    # Summary of the user input

    dcc.Store(id='estimates', storage_type='session'),
    # The simulations of the critical values, power etc.
    # This is a list of two json serialized pandas dataframes 1) the estimates and 2) their standard errors.

])

if __name__ == '__main__':
    app.run_server(debug=True)
