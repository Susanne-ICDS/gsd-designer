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

    <script>
      // Your web app's Firebase configuration
      // For Firebase JS SDK v7.20.0 and later, measurementId is optional
      var firebaseConfig = {
        apiKey: "AIzaSyDI-7W4G6OBk1e34TAQsnhDvmPQ5XbAQBo",
        authDomain: "gsdesigner-81c5e.firebaseapp.com",
        projectId: "gsdesigner-81c5e",
        storageBucket: "gsdesigner-81c5e.appspot.com",
        messagingSenderId: "53303168086",
        appId: "1:53303168086:web:dcb1085c9a9674a5fbfff7",
        measurementId: "G-10V9R0CZE4"
      };
      // Initialize Firebase
      firebase.initializeApp(firebaseConfig);
      firebase.analytics();
    </script>
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