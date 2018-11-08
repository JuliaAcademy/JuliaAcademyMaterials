ENV["GKSwstype"] = "100" #src

# # Tabular Data
#
# ## DataFrames
#
# The **`DataFrames`** package implements a tabular data format that behaves similarly to 
# a `data.frame` in R and a `pandas.DataFrame` in Python.
#
# We'll start by loading the **`RDatasets`** package, which provides a variety of sample 
# datasets.

using DataFrames, RDatasets

iris = dataset("datasets", "iris")  # load "iris" from R's "datasets" package

# ## Plotting columns of a DataFrame
#
# In the previous module we used the **`StatPlots`** package to make statistical visualizations.
# Another feature that **`StatPlots`** provides is a `@df`, which is used to conveniently
# plot the columns of a data table by referencing the names of the columns.

using StatPlots

@df iris scatter(:SepalLength, :SepalWidth, group=:Species, marker=:auto)
