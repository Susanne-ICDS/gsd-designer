import dash_html_components as html
import dash_bootstrap_components as dbc

spacing_variables = {'offset': 1, 'size': 10, 'int_input': 1, 'float_input': 2}

vub_blue = '#003399'
label = html.B


def my_jumbo_box(header, sub_header):
    return dbc.Row(
        dbc.Col(width={'offset': spacing_variables['offset'], 'size': spacing_variables['size']},
                children=[html.Br(),
                          html.H4(header, style={'color': 'white'}),
                          html.Label(sub_header, style={'color': 'white'}),
                          html.Br(),
                          html.Label(' ')]),
        style={'background-color': vub_blue})


table_style = {
    'css': [{'selector': '.dash-spreadsheet-container',
             'rule': 'border-radius: 6px; overflow: hidden;'}],
    'style_cell': {'fontSize': 16, 'font-family': 'sans-serif', 'text-align': 'left', 'padding': '12px'},
    'style_header': {'color': 'white', 'background-color': vub_blue},
    'style_data': {'color': '#343a40', 'background-color': 'white'}}

disabled_style_header = {'color': '#E8E8E8', 'background-color': '#808080'}
disabled_style_data = {'background-color': '#E8E8E8', 'color': '#808080'}
