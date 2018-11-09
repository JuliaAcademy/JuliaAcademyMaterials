ENV["GKSwstype"] = "100" #src

# # Working with Data
#
# > Data is the new oil. It’s valuable, but if unrefined it cannot really be used. It has to be changed into gas, plastic, chemicals, etc to create a valuable entity that drives profitable activity; so must data be broken down, analyzed for it to have value.
# - Clive Humbly, 2006
#
# # Tabular Data Structures
#
# 


# ## [DataFrames](https://github.com/JuliaData/DataFrames.jl)
# 
# <img src="http://juliadata.github.io/DataFrames.jl/latest/assets/logo.png" width=400>
#
# **`DataFrames`** is a popular tabular data package that we've already seen in the previous course modules.  Let's use the **`RDatasets`** package to load a dataset about diamond quality:

using DataFrames
import RDatasets

diamonds = RDatasets.dataset("ggplot2", "diamonds")

# We can get a subset of columns:

diamonds[[:Carat, :Cut, :Color]]

# ## [JuliaDB](https://github.com/JuliaComputing/JuliaDB.jl)
#
# <img src="https://user-images.githubusercontent.com/25916/36773410-843e61b0-1c7f-11e8-818b-3edb08da8f41.png" width=400>
#
# **`JuliaDB`** is similar to **`DataFrames`**, but with a number of key differences:
#
# - It has two data types for accessing data in different ways
# - It can work with data larger than computer memory
# - It integrates with [**`OnlineStats`**](https://github.com/joshday/OnlineStats.jl) to calculate statistics and models [out-of-core](https://en.wikipedia.org/wiki/External_memory_algorithm)

using JuliaDB


# ## Table Operations with [Query](https://github.com/queryverse/Query.jl)
# 
# The **`Query`** package describes itself as "Query almost anything in Julia".  It can be used with both **`DataFrames`** and **`JuliaDB`** as well as arrays, dictionaries, and more.  The basic syntax for querying "almost anything" is:
#
# ```
# q = @from <range variable> in <source> begin
#     <query statements>
# end
# ```
#
# 

using Query







#-----------------------------------------------------------------------# OLD
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
