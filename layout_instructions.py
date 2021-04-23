import dash_html_components as html

spacing_variables = {'offset': 1, 'size': 10, 'int_input': 1, 'float_input': 2}

vub_blue = '#003399'
label = html.B


def page_header(text): return html.H4(text, style={'color': 'white'})
def regular_text(text): return html.Label(text, style={'color': 'white'})


table_style = {
    'css': [{'selector': '.dash-spreadsheet-container',
             'rule': 'border-radius: 6px; overflow: hidden;'}],
    'style_cell': {'fontSize': 16, 'font-family': 'sans-serif', 'text-align': 'left', 'padding': '12px'},
    'style_header': {'color': 'white', 'background-color': vub_blue},
    'style_data': {'color': '#343a40'}}
