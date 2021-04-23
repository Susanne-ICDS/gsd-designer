import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
from dash_extensions import Download

# Global variables for this page only
# Preceding underscore '_' makes sure it cannot be imported to other pages
_default_accuracy = 10 ** -2
_min_accuracy = 10 ** -4
_max_accuracy = 10 ** -1

layout = html.Div([
    html.H4('Simulation'),
    html.P('Time to simulate the properties of your design'),

    dbc.Button('Evaluate', id='button', color='dark'),
    # This is the button that triggers the simulations

    html.Label('Output'),
    # A little loading GIF is shown as long as the simulations are running.
    dcc.Loading(id="loadingItem", type="default",
                children=[dbc.Alert("This is a primary alert", color="primary", id="status"),
                          html.Div(id='table')]),
    # The simulation results are shown in the div 'table'
    # The 'status' div is used to either invite the user to push the button,
    # show the number of performed simulations,
    # or show error messages in case of dumb user input.

    html.Label('Do you want a more detailed report?'),
    dbc.Button('Download CSV', id='csv_button', color='primary', outline=True),
    dbc.Button('Download excel', id='excel_button', color='dark'),
    html.Div('The excel file is more complete as it contains a tab showing your design input. ' +
             'The csv file only contains the output and its standard errors.'),
    Download(id="download"),

    html.Label('Relative tolerance and confidence level'),
    html.Div(id='explainAccuracy'),
    # This div shows the user a string explaining what the below parameters mean.
    # Since users are likely not familiar with numerical approximation terminology

    dcc.Input(id='relative-tolerance', placeholder='Relative tolerance', type='number', value=_default_accuracy,
              min=_min_accuracy, max=_max_accuracy, step=_min_accuracy, debounce=True),
    dcc.Input(id='CI', placeholder='Confidence level', type='number', value=0.95, min=0, max=1, debounce=True),
])