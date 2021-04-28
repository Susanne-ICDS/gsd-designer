import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_table
from dash.exceptions import PreventUpdate

import pandas as pd
import numpy as np

from statistical_parts.math_parts import t_test_functions
from statistical_parts.math_parts import one_way_functions

from layout_instructions import spacing_variables as spacing
from layout_instructions import label, table_style, regular_text

# Global variables for this page only
# Preceding underscore '_' makes sure it cannot be imported to other pages
_min_analyses = 2
_max_analyses = 7
_default_sample_step = 3
_min_groups = 2
_max_groups = 12

# Dictionary of tests for the dropdown menu
test_options = [{'label': 't-test', 'value': 'T'},
                {'label': 'one-way ANOVA', 'value': 'One-way'}]


# region definition of test objects
def TestObject(testName):
    """ Return the correct test object

    The used naming style is a bit deceptive since this is a function rather than a class.
    I chose this representation since it returns an object and is meant to be used as an object."""
    if testName == 'T':
        return TTest()
    if testName == 'One-way':
        return OneWay()


class BasicTest:
    """ Parent class for the various test objects """

    @classmethod
    def __init__(cls):
        """ Define the input for making layout """
        cls.VarGroups = html.Div([
            html.Br(),
            dbc.Row(dbc.Col(label('Number of experimental groups'),
                            width={'offset': spacing['offset'], 'size': spacing['size']})),
            dbc.Row(dbc.Col(dbc.Input(id='n_groups', placeholder='Groups', type='number', value=_min_groups + 1,
                                      min=_min_groups, max=_max_groups, step=1),
                            width={'offset': spacing['offset'], 'size': spacing['int_input']}))])

        cls.TwoGroups = html.Div(dbc.Input(id='n_groups', type='number', value=2), hidden=True)

        cls.Analyses = html.Div([
            dbc.Row(dbc.Col(label('Number of analyses'),
                            width={'offset': spacing['offset'], 'size': spacing['size']})),
            dbc.Row(dbc.Col(dbc.Input(id='n_analyses', placeholder='Analyses', type='number',
                                      value=_min_analyses, min=_min_analyses, max=_max_analyses, step=1),
                            width={'offset': spacing['offset'], 'size': spacing['int_input']}))])

    def tab_interim_analyses(self):
        return html.Div([
            self.Analyses,
            html.Br(),
            dbc.Row(dbc.Col(label('Sample size per group per analysis'),
                            width={'offset': spacing['offset'], 'size': spacing['size']})),
            dbc.Row(dbc.Col(dash_table.DataTable(id='sample_sizes', columns=[], editable=True, **table_style,
                                                 data=[{'analysis-{}'.format(i): _default_sample_step * (i + 1)
                                                        for i in range(_max_analyses)} for _ in range(_min_groups)]),
                            width={'offset': spacing['offset'], 'size': 'auto'}))])

    @staticmethod
    def resize_rows(n_groups, rows):
        """ Resize the rows of a dash table (without ids) to match the number of experimental groups. """

        n_groups = int(n_groups)
        if len(rows) < n_groups:
            return rows + [rows[-1] for _ in range(n_groups - len(rows))]
        return rows[:n_groups]

    @staticmethod
    def default_costs(n_analyses, n_groups, sample_sizes):
        """ Calculates the costs based on sample sizes.

        The cost at analysis i = the total sample size at analysis i."""

        n_analyses = int(n_analyses)
        n_groups = int(n_groups)
        # Sum sample sizes of all groups for each analysis
        sum_sample_sizes = {'analysis-{}'.format(i): sum(sample_sizes[j]['analysis-{}'.format(i)]
                                                         for j in range(n_groups)) for i in range(n_analyses)}
        return [sum_sample_sizes]

    @staticmethod
    def default_information_ratio(n_analyses, n_groups, sample_sizes):
        """ Calculates the information based on sample sizes.

        The information at analysis i =
        the total sample size at analysis i/ the total sample size at the final analysis. """

        n_analyses = int(n_analyses)
        n_groups = int(n_groups)
        # Sum sample sizes of all groups for each analysis
        sum_sample_sizes = [sum(sample_sizes[j]['analysis-{}'.format(i)] for j in range(n_groups))
                            for i in range(n_analyses)]
        # Divide by total sample size at final analysis
        return [{'analysis-{}'.format(i): sum_sample_sizes[i] / sum_sample_sizes[-1] for i in range(n_analyses)}]

    @staticmethod
    def check_sample_size(sample_sizes, n_analyses):
        """ Check if user input of the sample size table is complete and sensible. """

        if np.any([[sample_sizes[j]['analysis-{}'.format(i)] == '' for i in range(n_analyses)]
                  for j in range(len(sample_sizes))]):
            return True, 'Please fill in all cells for the sample size input table.'

        sample_sizes = np.array(pd.DataFrame(sample_sizes))
        sample_sizes = sample_sizes[:, :n_analyses]

        if np.any(sample_sizes != sample_sizes.astype(int)):
            return True, 'Sample sizes should be integers. Please check your input.'
        if np.any(sample_sizes < 2):
            return True, 'The sample size in each group should be at least 2. Please check your input.'
        if np.any(sample_sizes[:, 1:] < sample_sizes[:, :-1]):
            return True, 'The total sample size per group cannot decrease between analyses. Please check your input'

        return False, sample_sizes.astype(int)

    @staticmethod
    def update_test_parameter_input(**kwargs):
        """ By default there are no input fields that need to be changed based on other input.
        If there is, then the child class can overwrite this definition. """

        raise PreventUpdate

    @staticmethod
    def update_input_table(**kwargs):
        """ By default there are no input tables that need to be changed based on other input.
        If there is, then the child class can overwrite this definition. """

        raise PreventUpdate

    @staticmethod
    def test_params_to_dict(test_param_values, test_param_values_ids, test_param_data, test_param_data_ids):
        """ Transform the test specific input into a dictionary,
        so it can be passed to other functions properly. """

        part1 = {}
        part2 = {}

        if test_param_values is not None:
            part1 = {item['name']: test_param_values[i] for (i, item) in enumerate(test_param_values_ids)}

        if test_param_data is not None:
            for (i, item) in enumerate(test_param_data_ids):
                if test_param_data[i] is None:
                    part2[item['name']] = None
                else:
                    df = pd.DataFrame(test_param_data[i])
                    # The id column exists for clarity for the user.
                    # They play no role in the calculations and are therefore dropped.
                    if 'id' in df.columns:
                        part2[item['name']] = df.drop('id', axis=1).values.tolist()
                    else:
                        part2[item['name']] = df.values.tolist()

        return {**part1, **part2}

    def check_input(self, test_param_values, test_param_values_ids, test_param_data, test_param_data_ids, **kwargs):
        """ Put input in correct format and check if users wrote nonsense.

        Presumably this will be overwritten in every child class for specific test input. """

        test_parameters = self.test_params_to_dict(test_param_values, test_param_values_ids, test_param_data,
                                                   test_param_data_ids)
        return False, test_parameters

    @staticmethod
    def fixed_sample_size(**kwargs):
        return 'Secondary', 'The fixed sample size calculation has not yet been implemented for this test.'


class TTest(BasicTest):
    """ Test object for the independent samples t-test. """

    def tab_basic_design(self):
        """ The test specific input fields for the web page. """

        return html.Div([
            dbc.Row(dbc.Col(dbc.RadioItems(id={'type': 'test parameter', 'name': 'sides', 'form': 'value'},
                                           options=[{'label': 'one-sided', 'value': 'one'},
                                                    {'label': 'two-sided', 'value': 'two'}], value='one', inline=True),
                            width={'offset': spacing['offset'], 'size': spacing['size']})),

            self.TwoGroups,

            html.Br(),
            dbc.Row(dbc.Col([label('Pessimistic means and standard deviation')],
                            width={'offset': spacing['offset'], 'size': spacing['size']})),
            dbc.Row([
                dbc.Col([
                    html.Br(),
                    dash_table.DataTable(id={'type': 'test parameter', 'name': 'means', 'form': 'datatable'},
                                         columns=[{'id': 'mean-1', 'name': 'Mean group 1', 'type': 'numeric'},
                                                  {'id': 'mean-2', 'name': 'Mean group 2', 'type': 'numeric'}],
                                         data=[{'mean-1': 0, 'mean-2': 2}],
                                         editable=True, **table_style)],
                    width={'offset': spacing['offset'], 'size': 'auto'}),
                dbc.Col([
                    html.Br(),
                    dbc.Input(id={'type': 'test parameter', 'name': 'sd', 'form': 'value'},
                              placeholder='Standard deviation', type='number', min=0, step=0.000001)],
                    width={'size': spacing['float_input']})])
        ])

    def check_input(self, test_param_values, test_param_values_ids, test_param_data, test_param_data_ids, **kwargs):
        """ Put input in correct format and check if users wrote nonsense. """

        test_parameters = self.test_params_to_dict(test_param_values, test_param_values_ids, test_param_data,
                                                   test_param_data_ids)

        means = np.asarray(test_parameters['means'])

        if np.any(means == ''):
            return True, 'Please fill in all cells for the means input table.'
        if abs(means[:, 1] - means[:, 0]) < 10 ** -9:
            return True, 'Please fill in different means for both groups.'
        if 'sd' not in test_parameters.keys() or test_parameters['sd'] is None:
            return True, 'Please fill in a value for the standard deviation.'
        if test_parameters['sd'] < 10 ** -9:
            return True, 'Standard deviation cannot be zero. Please fill in a different value.'

        test_parameters['cohens_d'] = abs(means[:, 1] - means[:, 0])/test_parameters['sd']

        del test_parameters['means']
        del test_parameters['sd']

        return False, test_parameters

    def fixed_sample_size(self, alpha, beta, test_param_values, test_param_values_ids, test_param_data,
                          test_param_data_ids, **kwargs):
        problem, test_parameters = self.check_input(test_param_values, test_param_values_ids, test_param_data,
                                                    test_param_data_ids)
        if problem:
            return 'warning', test_parameters

        n, typeII = t_test_functions.give_fixed_sample_size(test_parameters['cohens_d'], alpha, beta,
                                                            test_parameters['sides'])

        return 'secondary', 'The required sample size for a fixed sample design is {} per group.'.format(n)

    @staticmethod
    def simulate_statistics(n_simulations, sample_sizes, hypothesis, test_parameters):
        """ Simulate the test statistics """

        if hypothesis == 'H0':
            return t_test_functions.simulate_statistics(n_simulations, sample_sizes, cohens_d=0,
                                                        sides=test_parameters['sides'])
        if hypothesis == 'HA':
            return t_test_functions.simulate_statistics(n_simulations, sample_sizes, **test_parameters)

    @staticmethod
    def give_exact(sample_sizes, alphas, betas, test_parameters):
        """ Return the exact values for the first analysis. """

        return t_test_functions.give_exact(sample_sizes, alphas, betas, **test_parameters)


class OneWay(BasicTest):
    """ Test object for the one-way ANOVA. """

    def tab_basic_design(self):
        """ The test specific input fields for the web page. """

        return html.Div([
            self.VarGroups,

            html.Br(),
            dbc.Row(dbc.Col(label('Pessimistic means per group and standard deviation'),
                            width={'offset': spacing['offset'], 'size': spacing['size']})),
            dbc.Row([dbc.Col(
                dash_table.DataTable(id={'type': 'test parameter', 'name': 'means', 'form': 'datatable'},
                                     columns=[{'id': 'id', 'name': '', 'editable': False},
                                              {'name': 'Expected mean', 'id': 'expected_means', 'editable': True,
                                               'type': 'numeric'}],
                                     data=[{'id': 'Group {}'.format(i+1), 'expected_means': 0}
                                           for i in range(_min_groups)], **table_style),
                width={'offset': spacing['offset'], 'size': 'auto'}),
                dbc.Col(dbc.Input(id={'type': 'test parameter', 'name': 'sd', 'form': 'value'},
                                  placeholder='Standard deviation', type='number', min=0, step=10**-6),
                        width=spacing['float_input'])])])

    @staticmethod
    def update_input_table(n_groups, rows, *args, **kwargs):
        """ Overwritten from the parent class. Make sure the number of input fields
        for the means match the number of experimental groups. """

        if rows is None or rows == []:
            raise PreventUpdate

        n_groups = int(n_groups)
        if len(rows) < n_groups:
            return rows + [{'id': 'Group {}'.format(i+1), 'expected_means': rows[-1]['expected_means']}
                           for i in np.arange(len(rows), n_groups)]
        return rows[:n_groups]

    def check_input(self, test_param_values, test_param_values_ids, test_param_data, test_param_data_ids, **kwargs):
        """ Put input in correct format and check if users wrote nonsense. """
        n_groups = kwargs['n_groups']

        test_parameters = self.test_params_to_dict(test_param_values, test_param_values_ids, test_param_data,
                                                   test_param_data_ids)

        if np.any(np.asarray(test_parameters['means']) == ''):
            return True, 'Please fill in all cells for the means input table.'

        if test_parameters['sd'] is None:
            return True, 'Nothing was entered for the standard deviation. Please check your input.'

        if test_parameters['sd'] == 0:
            return True, 'Standard deviation cannot be zero. Please check your input.'

        if test_parameters['means'] is None:
            return True, 'Please fill in a mean value for all experimental groups.'

        if np.asarray(test_parameters['means']).size != n_groups or \
                np.any(np.isnan(np.asarray(test_parameters['means']))):
            return True, 'Please fill in a mean value for all experimental groups.'

        if np.all(np.asarray(test_parameters['means']) == test_parameters['means'][0]):
            return True, 'At least one expected mean needs to be different from the rest. Please check your input.'

        return False, test_parameters

    @staticmethod
    def simulate_statistics(n_simulations, sample_sizes, hypothesis, test_parameters):
        """ Simulate the test statistics """

        if hypothesis == 'H0':
            return one_way_functions.simulate_statistics(n_simulations, sample_sizes,
                                                         means=np.zeros(sample_sizes.shape[0]),
                                                         sd=test_parameters['sd'])
        else:
            return one_way_functions.simulate_statistics(n_simulations, sample_sizes, **test_parameters)

    @staticmethod
    def give_exact(sample_sizes, alphas, betas, test_parameters):
        return one_way_functions.give_exact(sample_sizes, alphas, betas, **test_parameters)
# end region
