# # Data Viz
#
# Visualization is often the first step for any data analysis.  In this section we'll cover 
# Julia's best data viz tools.
#
# Note that Julia does not have a built-in plotting library.  There are other options available,
# but we will focus here on the fantastic **Plots** interface.
#
# # Plots.jl 
# ![](http://docs.juliaplots.org/latest/examples/img/lorenz.gif)
# **Plots** defines an interface for plotting data that works with multiple plotting libraries
# as the backend for creating the actual visualization.


# To begin, load the Plots package

using Plots 

# Input arguments can take many forms, but we'll start with plotting some random numbers.

x = 1:10
y = randn(10)
plot(x, y)

# The `x` values are implicitly set as `1:length(y)` if not provided.

plot(y)

# There are many attributes available to change via keyword arguments.

plot(y, seriestype = :scatter, markersize = 5, markercolor = :black)

# ## Series Types

# One of the more important keyword arguments to the `plot` function is `seriestype`, as it 
# determines how the input is turned into a plot.  For convenience, you can also call a 
# `seriestype` as a function :

plot(y, seriestype = :scatter)

#- 

scatter(y)

# ## Argument Aliases
# **Plots** supports "aliases" for keyword arguments for fast interactive creation of plots.  
# The idea for this is to help remove the need to constantly check the documentation for 
# the correct argument name.

plot(y, seriestype = :scatter)

#-

plot(y, st = :scatter)


# ## Magic Arguments

# One feature unique to **Plots** is that of magic arguments.  Certain keyword arguments act 
# like "magic" where by providing a tuple, Plots intelligently maps the items to the 
# corresponding plot element.  For example, by providing `marker = (5, .3, :auto)`, the 
# marker size is set as 5, the marker alpha as .3, and the marker shape is automatically 
# determined.

scatter(randn(10, 3), marker = (5, .3, :auto))

# # StatPlots.jl 
# The **StatPlots** package is an extension of **Plots** that provides utilities for 
# creating plots that are statistical in nature.  

using StatPlots

