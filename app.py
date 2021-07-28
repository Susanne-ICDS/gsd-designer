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