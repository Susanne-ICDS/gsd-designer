import dash
import dash_bootstrap_components as dbc

# Because different statistical test require to load different page layouts, the app needs to be defined in a different
# file. According to Dash documentation calling the app within the same file would lead to a loop or something
# app = dash.Dash(external_stylesheets=[dbc.themes.FLATLY])
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
    <!-- The core Firebase JS SDK is always required and must be listed first -->
    <script src="https://www.gstatic.com/firebasejs/8.6.1/firebase-app.js"></script>

    <!-- TODO: Add SDKs for Firebase products that you want to use
         https://firebase.google.com/docs/web/setup#available-libraries -->
    <script src="https://www.gstatic.com/firebasejs/8.6.1/firebase-analytics.js"></script>
    <script src="/__/firebase/8.6.1/firebase-performance.js"></script>

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