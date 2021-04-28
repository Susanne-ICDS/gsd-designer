import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
from dash_extensions import Download

from layout_instructions import spacing_variables as spacing
from layout_instructions import label, my_jumbo_box, regular_text, table_style

# Global variables for this page only
# Preceding underscore '_' makes sure it cannot be imported to other pages
_default_accuracy = 10 ** -2
_min_accuracy = 10 ** -4
_max_accuracy = 10 ** -1

layout = html.Div([
    my_jumbo_box('Simulations', 'Get your results'),
    dbc.Row(dbc.Col(width={'offset': spacing['offset'], 'size': spacing['size']},
                    children=[
                        html.Br(),
                        label('Results'), html.Br(),
                        regular_text('Click this button to simulate the properties of your design. The results remain '
                                     'even if you navigate away from this page, as long as you do not close your '
                                     'browser.'), html.Br(),
                        dbc.Button('Evaluate', id='button', color='primary')
                        ])),
    html.Br(),
    dbc.Row(dbc.Col(width={'offset': spacing['offset'], 'size': 'auto'},
                    children=[
                        # A little loading GIF is shown as long as the simulations are running.
                        dcc.Loading(id="loadingItem", type="default",
                                    children=[
                                        dbc.Alert(id="status", dismissable=True, is_open=False),
                                        html.Div(id='table')])])),

    # The simulation results are shown in the div 'table'
    # The 'status' div is used to either invite the user to push the button,
    # show the number of performed simulations,
    # or show error messages in case of dumb user input.
    dbc.Row(dbc.Col(width={'offset': spacing['offset'], 'size': spacing['size']},
                    children=[
                        html.Br(),
                        label('Download a more detailed report'), html.Br(),
                        regular_text('The csv file contains the unrounded estimates of the values above plus a few '
                                     'more, and each of their standard errors. The excel file has a tab with the same'
                                     'information as the csv file and an additional tab showing your design input.'
                                     ),
                        html.Br(),
                        dbc.Button('Download CSV', id='csv_button', color='primary', outline=True),
                        dbc.Button('Download excel', id='excel_button', color='primary', outline=True),
                        html.Br(),
                        html.Br(),
                        label('Relative tolerance and confidence level'),
                        html.Div(id='explain-accuracy'),
                        # This div shows the user a string explaining what the below parameters mean.
                        # Since users are likely not familiar with numerical approximation terminology
                        ])),
    dbc.Row([dbc.Col(width={'offset': spacing['offset'], 'size': spacing['float_input']},
                     children=[
                        dbc.Input(id='relative-tolerance', placeholder='Relative tolerance', type='number',
                                  value=_default_accuracy, min=_min_accuracy, max=_max_accuracy, step=_min_accuracy)]),
             dbc.Col(width=spacing['float_input'],
                     children=[
                         dbc.Input(id='CI', placeholder='Confidence level', type='number', value=0.95, min=0, max=1,
                                   step=10**-4)
                     ])]),
    Download(id="download")
])
