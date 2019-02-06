ENV["GKSwstype"] = "100" #src

# # Data Vizualization
#
# > The greatest value of a picture is when it forces us to notice what we never expected to see.
# > 
# > /- John Tukey
#
# Data visualization is used throughout the data science workflow.  It's used for everything from identifying simple relationships between variables to understanding how hyper-parameters affect model performance.  This course module will focus on exploratory data analysis using visualizations.
#
# Note that Julia does not have a built-in plotting library, but there are many to choose from:
#
# - [**`Gadfly`**](https://github.com/GiovineItalia/Gadfly.jl)
# - [**`GR`**](https://github.com/jheinen/GR.jl)
# - [**`PGFPlotsX`**](https://github.com/KristofferC/PGFPlotsX.jl)
# - [**`PlotlyJS`**](https://github.com/sglyon/PlotlyJS.jl)
# - [**`Plots`**](https://github.com/JuliaPlots/Plots.jl)
# - [**`PyPlot`**](https://github.com/JuliaPy/PyPlot.jl)
# - [**`UnicodePlots`**](https://github.com/Evizero/UnicodePlots.jl)
# - [**`Winston`**](https://github.com/JuliaGraphics/Winston.jl)
#
# If you are new to Julia, this may be an overwhelming number of options.  We highly recommend the **`Plots`** package, as it is a great option for both newcomers and power users.  This course module uses **`Plots`** and its related packages.

# # What is [Plots.jl](https://github.com/JuliaPlots/Plots.jl)?
#
# ![](https://raw.githubusercontent.com/JuliaPlots/PlotReferenceImages.jl/master/PlotDocs/index/lorenz_attractor.gif)
#
# **`Plots`** defines an interface for plotting that works with multiple plotting libraries "backends", including **`PyPlot`**, **`GR`**, and **`UnicodePlots`**.  It allows you to switch between backends, making it painless to go from an interactive javascript plot to something print-ready.  Let's get started with some of the core **`Plots`** concepts.

using Plots
gr()  # set the backend as GR

# The `plot` function has many different methods.  

p1 = plot(rand(10))             # y variable only
p2 = plot(-5:4, rand(10))       # x and y variables
p3 = plot(rand(10, 3))          # multiple y variables (each column)
p4 = plot(-5:4, rand(10, 3))    # x and multiple y variables

plot(p1, p2, p3, p4)            # join plots into subplots

# There are many attributes available to change via keyword arguments.

plot(rand(10), seriestype = :scatter, markersize = 5, markercolor = :black)

# ## Series Types 
#
# One of the most important keyword arguments to `plot` is `seriestype`, as it determines how the input data is turned into a plot.  For convenience, a `seriestype` also has a function of the same name:

y = rand(10)

scatter(y)  # same as plot(y, seriestype = :scatter)

# ## Argument Aliases
#
# **Plots** supports "aliases" for keyword arguments for fast interactive creation of plots.  You typically don't need to keep checking the documentation for the exact keyword.  Aliases are intuitive and you'll often be able to guess the correct syntax.

plot(y, markershape = :circle)

#-

plot(y, m = :circle)

# ## Magic Arguments
#
# *Magic arguments* is one of **`Plots`**' unique features.  Certain keyword arguments act like "magic" where by providing a tuple, Plots intelligently maps the items to the corresponding plot element.  For example, by providing `marker = (5, .3, :auto)`, the marker size is set as 5, the marker alpha as .3, and the marker shape is automatically determined.  Note that we can also use the alias `m` rather than `marker`:

scatter(randn(10, 3), m = (5, .3, :auto))

# ## Plotting from datasets
#
# Let's now use what we've learned about **`Plots`** to explore the `mtcars` dataset, extracted from a 1974 issue of of *Motor Trend* Magazine.  The variables included are:
#
# - `Model`: Make and model of the car
# - `MPG`: Miles per gallon
# - `Cyl`: Number of Cylinders
# - `Disp`: Displacement (cubic inches)
# - `HP`: Gross horsepower
# - `DRat`: Rear axle ratio
# - `WT`: Weight (in 1000s lbs.)
# - `QSec`: Quarter mile time
# - `VS`: Engine type, V-shaped (0) or straight (1)
# - `AM`: Transmission, Automatic (0) or manual (1)
# - `Gear`: Number of forward gears
# - `Carb`: Number of carburetors

using RDatasets

mtcars = dataset("datasets", "mtcars")

# First, let's `describe` the data.

describe(mtcars)

# What's the relationship between miles per gallon and horsepower?  

scatter(mtcars.MPG, mtcars.HP, xlab="MPG", ylab = "HP")

# The `dataframe.column` syntax isn't always convenient for making complex visualizations.  The **`StatsPlots`** package adds a number of features to **`Plots`** for creating statistical graphics.  One of its main features is the `@df` macro that allows you to reference columns of a data table by name inside of a call to `plot`

using StatsPlots

@df mtcars corrplot([:MPG :Cyl :HP :AM], bins=25, grid=false)

# Now let's look at the distribution of `MPG` across `Cyl` using a [violin plot](https://en.wikipedia.org/wiki/Violin_plot).

@df mtcars violin(:Cyl, :HP, legend = :topleft, xlab="Cyl", ylab = "HP")

# **`Plots`** objects can also be mutated with additional data series.  Let's now overlap [box plots](https://en.wikipedia.org/wiki/Box_plot) of the same data, using the `alpha` keyword argument to make the box plots transparent.

@df mtcars boxplot!(:Cyl, :HP, alpha = .5)