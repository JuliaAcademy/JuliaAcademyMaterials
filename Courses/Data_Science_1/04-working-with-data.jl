ENV["GKSwstype"] = "100" #src

# # Working with Data
#
# > Data is the new oil. It’s valuable, but if unrefined it cannot really be used. It has to be changed into gas, plastic, chemicals, etc to create a valuable entity that drives profitable activity; so must data be broken down, analyzed for it to have value.
# - Clive Humbly, 2006
#
# There are several great packages for working with tabular data in Julia.  We have already seen a little bit of **`DataFrames`**, but we will focus on **`JuliaDB`** in this course module.  Many of the reasons one would choose **`JuliaDB`** instead of **`DataFrames`** (and vice versa) are personal preferences.  Most packages that integrate with tabular data, such as **`StatPlots`**, work equally well with both, so your choice mainly depends on the slight differences in the interfaces for filtering and selecting data.  However, **`JuliaDB`** does have some key differences compared to **`DataFrames`**:
#
# - It has two data types for accessing data in different ways.
# - It can run distributed operations and work with data larger than computer memory.
# - It integrates with [**`OnlineStats`**](https://github.com/joshday/OnlineStats.jl) to calculate statistics and models [out-of-core](https://en.wikipedia.org/wiki/External_memory_algorithm).
#
# # [JuliaDB](https://github.com/JuliaComputing/JuliaDB.jl)
#
# <img src = "https://user-images.githubusercontent.com/25916/36773410-843e61b0-1c7f-11e8-818b-3edb08da8f41.png" width=600>
#
# We will cover the basic functionality of **`JuliaDB`** through an example.  Let's use the **`RDatasets`** package to load the `diamonds` dataset, a collection of measurements on the quality of ~54,000 diamonds.  The variables include:
#
# - Carat: weight of the diamond
# - Cut: quality of the cut (Fair, Good, Very Good, Premium, Ideal)
# - Color: diamond color, from F to D (worst to best)
# - Clarity: categorization of clarity with values (worst to best) I1, SI2, SI1, VS2, VS1, VVS2, VVS1, IF
# - Depth: total depth percentage = Z / mean(X, Y)
# - Table: width of diamond's top relative to widest point
# - Price: price in $US
# - X: length
# - Y: width
# - Z: depth
#
# The **`RDatasets`** package loads datasets as a `DataFrame`, but fortunately the data structures provided by **`DataFrames`** and **`JuliaDB`** can be easily converted back and forth.  A `DataFrame` can be converted to a `NextTable` (one of the core **`JuliaDB`** tabular data structures) with the `table` function.

using JuliaDB
import RDatasets

df = RDatasets.dataset("ggplot2", "diamonds")

t = table(df)

# ## Selecting Columns
#
# **`JuliaDB`** has many methods for selecting columns via the `select` function:

select(t, 1)  # select the column as position 1 (:Carat)
select(t, :Carat) # select the column with the given name

#-

select(t, (1,7))  # select the columns at positions 1 and 7 (:Carat and :Price)
select(t, (:Carat, :Price))  # select the columns :Carat and :Price
select(t, r"(Carat|Price)") # select the columns that match a regular expression

# ## Selecting Rows
#
# Rows are selected based on some criteria with the `filter` function.  The first argument is a function which is then applied to whatever is selected by the `select` keyword argument.  Any valid selection via the `select` function (Integers, column names as Symbols, Regexes, etc.) can be used.  
#
# Note that if you select a single column, the function is applied to a Vector.  If the selection retrieves a table, the function is applied to the rows of the table as NamedTuples.

filter(x -> x > .7, t; select = :Carat)  # function applied to Vector

#-

filter(x -> x.Carat > .7, t; select = r"(Carat|Price)")  # function applied to NamedTuples of rows

# ## Grouping
#
# A common table operation is to summarize a column across the groups in a different column.  With the `groupby` function , here we are calculating the average `:Price` for each level of `:Cut`.

using Statistics

groupby(mean, t, :Cut; select = :Price)

# It is necessary for groupby operations to allocate temporary vectors (one per group).  What if we could perform a groupby operation without allocating?  First consider the `reduce` function, which applies a pairwise function to the rows of a table.

reduce(+, t; select=:Price)  # Sum of prices for all diamonds in the dataset

# The `groupreduce` function will apply a pairwise function across the groups of a different variable, similar to `groupby`:

groupreduce(+, t, :Cut; select = :Price)

# The drawback is that the function argument for `reduce` and `groupreduce` must work on two items at a time.  To help alleviate this requirement, **`JuliaDB`** integrates with [**`OnlineStats`**](https://github.com/joshday/OnlineStats.jl), a package that provides parallelizable single-pass algorithms for statistics.  `OnlineStat` objects can be provided to both `reduce` and `groupreduce`, which opens up a wide variety of statistical calculations without the need for temporary allocations.

using OnlineStats

reduce(Sum(Int), t; select = :Price)

#-

groupreduce(Sum(Int), t, :Cut; select = :Price)

# Many `OnlineStat` objects are also plottable with **`Plots`**.  Here we will plot a histogram with the `Hist` stat.  We can see that diamonds are often cut down to be a "nice" carat size.

using Plots

o = reduce(Hist(0:.01:1), t; select = :Carat)

plot(o)
