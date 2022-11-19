import dash
import dash_bootstrap_components as dbc

# According to Dash documentation in order to load multiple page layouts the app needs to be defined in a separate file.
# So that is what this is. The html code required for Google Firebase is added here.

dash_app = dash.Dash(__name__, suppress_callback_exceptions=True, title='GSDesigner')
server = dash_app.server

dash_app.index_string = """
<!doctype html>
<html>
<head>
    {%metas%}
    <title>{%title%}</title>
    {%favicon%}
    {%css%}
</head>
<body>
    <script src="firebase_script.js"></script>
    
    {%app_entry%}
    <footer>
        {%config%}
        {%scripts%}
        {%renderer%}
    </footer>
    <div></div>
</body>
</html>
"""