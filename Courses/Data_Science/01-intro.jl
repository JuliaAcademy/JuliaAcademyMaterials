ENV["GKSwstype"] = "100" #src (hack so GR doesn't throw a fit)

# # A Primer on Data Science in Julia
#
# > [T]he key word in data science is not “data”; it is “science”.
#
# - Jeff Leek
# 
# # What is Data Science?
#
# The term **data science** is a relatively new term used to describe a variety of cross-discipline tasks that cover statistics, mathematics, computer science, data visualization, and information science.  Because of the wide variety of tasks involved, the role of a "data scientist" may differ greatly from business to business (or even within a single company).  At a high level, data science is the collection of activities that help one move upword in the *DIKW Pyramid*, shown below:
#
# ![](https://upload.wikimedia.org/wikipedia/commons/thumb/0/06/DIKW_Pyramid.svg/494px-DIKW_Pyramid.svg.png)
#
# **How do we get from data to knowledge/wisdom?**
#
# Movement up the pyramid happens in incremental steps.  Sometimes you'll have an "aha!" moment and make a leap in understanding, but that is typically due to the previous information you've gathered.  To aid with the task of incremental knowledge-building, the data science workflow is made up of the following steps:
#
# 1. Define Your Objective
# 2. Explore Your Data
# 3. Model Your Data
# 4. Explore Your Model
# 5. Communicate Your Results
#
# The modules in this course aim to assist with the above steps:
#
# - Getting Started
# - Data Visualization 
# - Working with Data Tables
# - Unsupervised Learning
# - Supervised Learnings
# - Communicating Data Science






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

# ## StatsBase
#
# The **`StatsBase`** package offers more statistical techniques beyond what 
# **`Statistics`** has.  We'll load the **`Plots`** package as well.

using StatsBase

quantile(y)










#-----------------------------------------------------------------------#
#-----------------------------------------------------------------------# OLD

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
# This module introduced you to the **`Statistics`** and **`StatsBase`** packages.  
# The next modules will cover more specific data science tasks such as fitting regressions,
# working with datasets, and data visualization.
