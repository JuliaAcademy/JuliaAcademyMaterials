ENV["GKSwstype"] = "100" #src (hack so GR doesn't throw a fit)

# # Getting Started with Data Science in Julia

println("Hello, World!")

# # Basic Statistics
#
# Our introduction to the world of data science will begin with basic statistical operations.
# The **`Statistics`** module, from Julia's standard libarary, provides many of the core 
# statistical machinery, such as means, variances, quantiles, etc.

using Statistics

x = randn(100, 4)

mean(x)

# Above we calculated the mean for every value in the Matrix y.  If instead we want the mean 
# over a specific dimension of the matrix (e.g. mean of each column), we can specify the 
# `dims` keyword argument.

mean(x, dims=1)

# There are many other operations besides `mean` that are provided in the **`Statistics*``
# module.  We can examine all the exported names of a module with the `names` function:

names(Statistics)

#-

cor(x)

# ## [StatsBase](https://github.com/JuliaStats/StatsBase.jl)
#
# The **`StatsBase`** package offers more statistical techniques beyond what 
# **`Statistics`** has.

using StatsBase

quantile(x[:, 1])

# ## Random Sampling
#
# [Random Sampling](https://en.wikipedia.org/wiki/Sampling_(statistics)) plays an integral part in many data science tasks, such as:
#
# - Splitting data into multiple data sets for [cross validation](https://en.wikipedia.org/wiki/Cross-validation_(statistics)).
# - Subsampling a large dataset to something more manageable.
# - Running statistical simulations.
# - [Statistical Bootstrap](https://en.wikipedia.org/wiki/Bootstrapping_(statistics))
#
# **`StatsBase`** has the `sample` function to factilitate sampling [with or without replacement](https://en.wikipedia.org/wiki/Sampling_(statistics)#Replacement_of_selected_units).  

y1 = sample(1:20, 20, replace=true)  # replaces units after being selected

#-

y2 = sample(1:20, 20, replace=false)  # DOES NOT replace units after being selected

# Next we'll use the [`countmap`](http://juliastats.github.io/StatsBase.jl/latest/search.html?q=countmap) function to get the count of each unique value in our samples.  Notice that when we do sampling *without* replacement, the values will only appear one time.  When sampling *with* replacement, any given selected item is then placed back into the pool of possible items to select from.

countmap(y1)

#- 

countmap(y2)

# This is just a small sample (pun intended) of the features in **`StatsBase`**.  For a more complete look of what is possible, check out the [documentation](http://juliastats.github.io/StatsBase.jl/latest/index.html).