# # Getting Started with Data Science in Julia
#
# Here we will cover some basic statistics functionality.  While this material is introductory,
# it assumes you are already familiar with Julia.  If you don't understand the code examples 
# yet, that's okay!  You may want to start with our "Introduction to Julia" course.
#
# ## Summary Statistics
#
# Basic statistics functionality is in the **`Statistics`** module, which is a part of Julia's 
# standard library.  To use common functions like `mean`, `var`, `cov`, and `quantile`, 
# first load the **`Statistics`** package.

using Statistics

y = randn(100)

mean(y)

# It's often the case that each column or row of a matrix represents a different variable
# and you would like to get the mean/var/etc. of each variable separately.  You can specify
# the keyword argument `dims` to apply the function over a specific dimension.

x = randn(5, 5);  # Note: use ';' to suppress output!

#- 

mean(x, dims=1)  # mean of each column

#-

mean(x, dims=2)  # mean of each row

# ## StatsBase
#
# The **`StatsBase`** package offers more statistical techniques beyond what 
# **`Statistics`** has.  We'll load the **`Plots`** package as well so we can plot the 
# histograms.

using StatsBase, Plots 

# ### StatsBase: Histograms
#
# Histograms are a common method of visualizing distributions.  Histograms split the data 
# into bins (half-closed intervals) that keep track of how many observations are contained 
# within.

y = randn(10^5)

h = fit(Histogram, y; closed = :left)  # `closed = :left` makes bins take the form [a, b)

plot(h)

# ### StatsBase: Random Sampling
#
# Random Sampling plays an integral part in many data science tasks, such as:
#
# - Splitting data into training and test sets.
# - Subsampling a large dataset to something more manageable.
# - Running statistical simulations.
#
# **`StatsBase`** introduces the `sample` function for sampling from data with or without 
# replacement.

sample(1:3, 10; replace = true)  # same as rand(1:3, 10)

sample(1:20, 10; replace = false)

# ### StatsBase: Counting Unique Values
#
# The `countmap` function maps each unique value of a dataset to the number of times it 
# occurs.

y = sample(1:20, 10; replace = false)

countmap(y)

#-

y = sample(1:3, 10; replace = true)

countmap(y)


# ## What's Next?
#
# TODO
