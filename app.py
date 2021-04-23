import dash
import dash_bootstrap_components as dbc

# Because different statistical test require to load different page layouts, the app needs to be defined in a different
# file. According to Dash documentation calling the app within the same file would lead to a loop or something
# app = dash.Dash(external_stylesheets=[dbc.themes.FLATLY])
app = dash.Dash(__name__, suppress_callback_exceptions=False)
server = app.server
