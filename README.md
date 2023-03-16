# gsd-designer

The purpose of this app is to aid researchers in the design of group sequential designs (GSD), featuring either t-tests, one-way ANOVA's, or Mann-Whitney-Wilcoxon test (aka Wilcoxon-Mann-Whitney, Mann-Whitney U, Wilcoxon sum rank test). The app was designed with small to medium sample sizes in mind, but will work for large sample sizes as well.

A tutorial of a basic example can be found as a pdf file in the github folder.

This app was created in python with dash (https://dash.plotly.com/). The implementation of the statistics can be found in the folder 'statistical_parts'. The other files and folders are for building the user interface, processing the interactions with the user interface, etc.

If you run the app locally, you can easily add custom error spending functions in the file 'statistical_parts\error_spending.py'
