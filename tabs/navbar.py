import dash_bootstrap_components as dbc

bar = dbc.NavbarSimple(
    [dbc.NavItem(dbc.NavLink("Basic design", id='tab1', href="basic-design")),
     dbc.NavItem(dbc.NavLink("Interim analyses", id='tab2', href="interim-analyses")),
     dbc.NavItem(dbc.NavLink("Error spending", id='tab3', href="error-spending")),
     dbc.NavItem(dbc.NavLink("Simulation", id='tab4', href="simulation")),
     dbc.NavItem(dbc.NavLink("Effect size estimation", id='tab5', href="effect-size-CI"))],
    brand="GSDesigner",
    brand_href="basic-design")
