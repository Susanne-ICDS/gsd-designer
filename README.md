# gsd-designer

The purpose of this app is to aid researchers in the design of group sequential designs (GSD) with small sample sizes, featuring either t-tests or one-way ANOVA's. While the app was specifically developped for preclinical research, it can be used in any context for GSDs with small sample sizes as long as the conditions for the t-test respectively 
one-way ANOVA are met.

A tutorial of a basic example can be found as a pdf file in the github folder.

This app was created in python with dash (https://dash.plotly.com/). The implementation of the statistics can be found in the folder 'statistical_parts'. The other files and folders are for building the user interface, processing the interactions with the user interface, etc.

If you run the app locally, you can easily add custom error spending functions in the file 'statistical_parts\error_spending.py'
To add a new statistical test, you need to add an object to 'statistical_parts\statistical_test_objects.py'. In theory you can add all necessary code there, but to keep the code clean and legible, it is nicer to put the mathematical formulas in a separate file in \math_parts\.