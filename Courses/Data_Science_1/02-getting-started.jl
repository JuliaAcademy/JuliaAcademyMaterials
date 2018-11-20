ENV["GKSwstype"] = "100" #src

# # Getting Started with Data Science in Julia
#
# > [A]lways bite the bullet with regard to understanding the basics of the data first before you do anything else...
#
# - Caitlin Smallwood, Vice President of Science and Algorithms at Netflix

println("Hello, Data Science World!")

# # What does your data look like?
#
# Before you can solve complex problems with data, you should have a firm grasp on what your data looks like.  Are your variables continuous or categorical?  What do the distributions look like?  Are there missing observations?  Are variables correlated?  All of these questions arise in the data science workflow, so it's helpful to know the answers from the start of your analysis.

# Let's start with loading the [**`Statistics`**](https://docs.julialang.org/en/latest/stdlib/Statistics/) package, which provides basic statistical operations like means, variances, etc.

using Statistics

# You can examine the items that a package exports with the `names` function.

names(Statistics)

# To start, let's try some of the functions from **`Statistics`** on simulated data.

x = randn(100, 3)

mean(x, dims=1)  # calculate over first dimension (column means)

#-

cor(x)  # correlation matrix  

# For a more realistic example, let's load some data from the **`RDatasets`** package, which has a large collection of datasets that get loaded as a `DataFrame`.  

# The [`"iris"` dataset](https://en.wikipedia.org/wiki/Iris_flower_data_set) is a collection of measurements for three different species of the iris flower: Iris Setosa, Iris Virginica, and Iris Versicolor (shown below).  The measurements consist of length and width of the petal and sepal (part underneath the flower).

# <img src="https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg" width=400>

using RDatasets

iris = dataset("datasets", "iris")

# DataFrames are discussed in another course module.  For now, we'll just use the fact that data vectors from the DataFrame can be selected with `mydf.mycol`:

iris.SepalLength

# What is the average sepal width accross all three species?

mean(iris.SepalWidth)

# Maximum and minimum?

@show minimum(iris.SepalWidth)
@show maximum(iris.SepalWidth)
extrema(iris.SepalWidth)

# Is petal width correlated with petal length?

cor(iris.PetalWidth, iris.PetalLength)

# While we could examine each column separately, a much quicker way to summarize our variables is with the `describe` function, which creates a new DataFrame where each row contains 

describe(iris)


# # Random Sampling
#
# <img src="https://upload.wikimedia.org/wikipedia/commons/b/bf/Simple_random_sampling.PNG" width=400>
#
# [Random Sampling](https://en.wikipedia.org/wiki/Sampling_(statistics)) plays an integral part in many data science tasks, such as:
#
# - Splitting data into multiple datasets for [cross validation](https://en.wikipedia.org/wiki/Cross-validation_(statistics)).
# - Subsampling a large dataset to something more manageable.
# - Running statistical simulations.
# - [Statistical Bootstrap](https://en.wikipedia.org/wiki/Bootstrapping_(statistics))
#
# The **`StatsBase`** package has the `sample` function to factilitate sampling [with or without replacement](https://en.wikipedia.org/wiki/Sampling_(statistics)#Replacement_of_selected_units).
# Imagine you had a bag with 20 numbers in it.
# Sampling with replacement means that you put the number you drew back into the bag before the next draw.
# Just like rolling a dice, each draw is _independent_ and a given number might get drawn again.
# On the other hand, if you were to _not_ put the drawn number back in the bag, that would be representative of sampling without replacement.
# Each draw is _dependent_ upon all previous draws as those numbers are no longer available for selection.
# For bootstrapping and statistical simulations, it's typical to use sampling with replacement.
# While the two behave nearly the same if the sample is a very small fraction of the overall population, it becomes important to sample without replacement if the sample is large or if using uneven importance weights.

using StatsBase

y1 = sample(1:20, 20, replace=true)  # replaces units after being selected

#-

y2 = sample(1:20, 20, replace=false)  # DOES NOT replace units after being selected

# Next we'll use the [`countmap`](http://juliastats.github.io/StatsBase.jl/latest/search.html?q=countmap) function to get the count of each unique value in our samples.  Notice that when we do sampling *without* replacement, the values will only appear one time.  When sampling *with* replacement, any given selected item is then placed back into the pool of possible items to select from.

countmap(y1)

#- 

countmap(y2)

# This is just a small sample (pun intended) of the features in **`StatsBase`**.  For a more complete look of what is possible, check out the [documentation](http://juliastats.github.io/StatsBase.jl/latest/index.html).




# # Parametric Distributions
#
# The [**`Distributions`**](https://github.com/JuliaStats/Distributions.jl) package provides an interface for working with probability distributions.  The full documentation is [here](https://juliastats.github.io/Distributions.jl/stable/).  

# Here we'll also load the **`StatPlots`** package (more on this in the next course module) to visualize the probability distributions.

using Distributions, StatPlots

plot(Normal(), label = "Normal(0, 1)")

#-

plot!(Gamma(5, 1), label = "Gamma(5, 1)")

# The package defines a number of functions that together create a consistent "grammar" for discussing distributions:
#
# - Probability density function: `pdf`
# - Cumulative distribution function: `cdf`
# - etc.


d = Normal()

pdf(d, 0), cdf(d, 0), mean(d), var(d), quantile(d, .5), mode(d)

#-

d = Gamma(5, 1)

pdf(d, 0), cdf(d, 0), mean(d), var(d), quantile(d, .5), mode(d)

# As an example, let's write a function that uses [Newton's Method](https://en.wikipedia.org/wiki/Newton%27s_method) that can find a given quantile for any continuous univariate distribution.  Newton's method attempts to find the root for a function $f$ by performing iterations of the form:
#
# $$\theta^{(t)} = \theta^{(t-1)} - \frac{f(\theta^{(t-1)})}{f'(\theta^{(t-1)})}.$$
#
# For quantiles, we want to find the root of the function $F(\theta) - q$ where $F$ is the cumulative density function of the distribtuion and $q \in (0, 1)$.  Using **`Distributions`**, this looks something like

function myquantile(d::Distribution, q::Number)
    θ = mean(d)
    for i in 1:20
        θ -= (cdf(d, θ) - q) / pdf(d, θ)  # θ = θ - (F(θ) - q) / F'(θ)
    end
    θ
end

# Does our `myquantile` function work as expected?  Let's try out our function on several distributions.

d = Normal()
myquantile(d, .5), quantile(d, .5)

#-

d = Gamma(4,3)
myquantile(d, .7), quantile(d, .7)

# The above example shows off the power of generic functions.  Instead of hard-coding the distribution (as would be necessary in R), we can write functions in terms of an arbitrary distribution (without extra effort).  This gives us a lot of flexibility for tasks such as writing [Gibbs Samplers](https://en.wikipedia.org/wiki/Gibbs_sampling) that can swap out distributions with ease.


# # Missing Data
#
# TODO
