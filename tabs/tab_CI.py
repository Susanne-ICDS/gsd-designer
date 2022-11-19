import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc

from layout_instructions import spacing_variables as spacing
from layout_instructions import label, my_jumbo_box, regular_text, table_style

# Global variables for this page only
# Preceding underscore '_' makes sure it cannot be imported to other pages
_default_accuracy = 10 ** -2
_min_accuracy = 10 ** -4
_max_accuracy = 10 ** -1

layout = html.Div([
    my_jumbo_box('Estimates and confidence intervals', 'Analysis after termination of the experiment'),
    dbc.Row(dbc.Col(width={'offset': spacing['offset'], 'size': spacing['size']},
                    children=[
                        html.Br(),

                        label('Used design'), html.Br(),
                        regular_text('Select the design you used. The ids in the dropdown menu refer to the models '
                                     'evaluated on the previous page. Note that if you entered all the values, you '
                                     'still need to simulate the critical bounds before you can select them to '
                                     'evaluate the CI.'), html.Br(),
                        dbc.Select(id="used_model", options=[]), html.Br(),
                        html.Br(),
                        label('Number of analyses and test statistic at termination')
                        ])),
    dbc.Row([dbc.Col(width={'offset': spacing['offset'], 'size': spacing['float_input']},
                     children=[
                        dbc.Input(id='n_termination', placeholder='Analysis number', type='number',
                                  min=0, max=0, step=1)]),
             dbc.Col(width=spacing['float_input'],
                     children=[
                         dbc.Input(id='result_statistic', placeholder='Test statistic', type='number', step=10**-4)
                     ])]),
    html.Br(),

    dbc.Row(dbc.Col(width={'offset': spacing['offset'], 'size': spacing['size']},
                    children=[label('Confidence level of the interval')])),

    dbc.Row(dbc.Col(width={'offset': spacing['offset'], 'size': spacing['float_input']},
                    children=[
                         dbc.Input(id='ci_confidence', placeholder='Confidence level', type='number', value=0.9, min=0,
                                   max=1, step=10**-4)])),
    html.Br(),

    dbc.Row(dbc.Col(width={'offset': spacing['offset'], 'size': spacing['size']},
                    children=[label('Estimate and confidence interval of the effect size'), html.Br(),
                              html.Br(),
                              dbc.Button('Evaluate', id='ci_button', color='primary')])),

    html.Br(),
    # A little loading GIF is shown as long as the simulations are running.
    dcc.Loading(id="loadingItemCI", type="default",
                children=[dbc.Row(dbc.Col(width={'offset': spacing['offset'], 'size': 'auto'},
                                          children=[html.Div(id='effect_estimate',
                                                             children=['If the experiment terminated at the first '
                                                                       'analysis, this is just the classical estimate '
                                                                       'and CI'])]))]),

    # The simulation results are shown in the div 'table'
    # The 'status' div is used to either invite the user to push the button,
    # show the number of performed simulations,
    # or show error messages in case of dumb user input.
    dbc.Row(dbc.Col(width={'offset': spacing['offset'], 'size': spacing['size']},
                    children=[
                        html.Br(),
                        label('Relative tolerance and maximum iterations')
                    ])),
    dbc.Row([dbc.Col(width={'offset': spacing['offset'], 'size': spacing['float_input']},
                     children=[
                        dbc.Input(id='rel-tol-CI', placeholder='Relative tolerance', type='number',
                                  value=_default_accuracy, min=_min_accuracy, max=_max_accuracy, step=_min_accuracy)]),
            dbc.Col(width={'size': 'size'}, children=[regular_text(', 10^')], align="end"),
            dbc.Col(width={'size': spacing['int_input']},
                    children=[
                        dbc.Input(id='max_iter', placeholder='', type='number', value=3, min=2, max=5, step=1)])
            ])
])
