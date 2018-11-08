ENV["GKSwstype"] = "100" #src

# # Density Estimation
#
# Understanding the shape of your data is important for any data analysis.
# 
# - Do you have continuous data? Categorical data? Both?  
# - Are variables correlated?
# - What insights can you gain from univariate and bivariate data summaries?
#
# There are many ways to estimate and visualize the distribution of data.  In this module,
# we'll cover: TODO
#
# ## Empirical Density Estimation
#
# ### Empirical Cumulative Density Function (ECDF)
#
# The **`StatsBase`** package provides the `ecdf` function, which returns a function that 
# can calculate the ECDF evaluated at a point, defined as 
#
# $$f_{\text{ECDF}}(x) = \frac{\text{# values less than or equal to x}}{\text{total # values}}.$$
#
# In other words, the number returned by the ECDF is the probability that a random sample 
# from the dataset is less than or equal to the argument.

using StatsBase, Plots

y = 1:10

f = ecdf(y)

plot(f, 0, 11, label="", xlab="x", ylab = "P(data <= x)")


# ### Univariate Data
#
# First we will look at how to estimate/plot univariate distributions of data. 
# As we saw in the intro module, we can plot histograms of continuous datafrom the 
# **`StatsBase`** package.

using StatsBase, Plots

y = randn(10_000)

h = fit(Histogram, y; closed = :left)

plot(h)

# Alternatively, we can create the plot with the `histogram` function from ``*Plots*``.

histogram(y)

# #### Kernel Density Estimation (KDE)
#
# Kernel Density Estimation (KDE) is a method of estimating the probability density function
# by averaging together a kernel function applied to neighboring points.  A KDE with kernel 
# $K$ and bandwidth parameter $h>0$ is a function of data points $(x_1,\ldots,x_n)$ that looks like 
#
# $$\hat f(x) = \frac{1}{nh} \sum_{i=1}^n K\left(\frac{x-x_i}{h}\right).$$
#
# Here are some examples of kernel functions:
#
# ![](https://user-images.githubusercontent.com/8075494/30523575-acd48de2-9bb1-11e7-8f0f-3ce2ab09c713.png)
#
# With the **`KernelDensity`** package, let's plot some KDEs with a variety of bandwidths
# using the default (Normal) kernel.  Note that the bandwidth choice has a very big effect
# on the shape of the estimated density.

using KernelDensity, StatPlots

k = kde(y)

p = plot(k, label = "Bandwidth: Auto", w=2)

for bw in .1:.2:.9
    plot!(p, kde(y; bandwidth = bw), label = "Bandwidth: $bw", w=2)
end

histogram!(p, y, alpha = .1, normed=true)


# #### Averaged Shifted Histograms (ASH)
#
# The Averaged Shifted Histogram (ASH) estimator is a bit of a misnomer; It is essentially
# kernel density estimation performed on a fine-partition (many small bins) histogram.  
#
# ASH has an advantage over KDEs in terms of performance since the density is calculated 
# over the bins instead of all observations.  Similarly, ASH has an advantage over histograms
# in that smoothing the density over many small bins gives a more fine-grained view of the
# distribution.

using AverageShiftedHistograms

a = ash(y)

plot(a)

# The line is a bit noisy, but we can increase the amount of smoothing with parameter `m`

ash!(a; m = 15)

plot(a)

# But make sure you don't oversmooth!

ash!(a; m = 50)
plot(a)

# Let's compare all three of our density estimates.

plot(ash!(a, m = 15), hist=false)
histogram!(y, normed=true, alpha=.2, label = "Histogram")
plot!(k, label = "KDE")



# #### Categorical Data

using OnlineStats 

y = rand([1,2,2,3,3,3,4,4,5], 10^5)

o = fit!(CountMap(Int), y)

plot(o)


# ### Bivariate Relationships
#
# So far, we've only looked at univariate distributions.  However, we can also use 
# Histograms, KDEs, and ASH estimators on bivariate data.

x = randn(10^4)
y = x .+ randn(10^4)

h = fit(Histogram, (x,y); closed = :left)
k = kde((x,y))
a = ash(x, y)

plot(
    plot(h; title="Histogram"), 
    plot(k; title="KDE"), 
    plot(a; title = "ASH"); 
    layout=3
)






# ## Parametric Density Estimation
#
# Suppose you already know some things about a dataset's distribution or you want to see
# how well the data fits a specific distribution.  Here we'll use the **`Distributions`** 
# package to fit data to distributions using maximum likelihood estimation (MLE).  We 
# also need the **`StatPlots`** package for plotting the distributions.

using Distributions

# In **`Distributions`**, each distribution is its own type, and there are a variety of
# methods that work for any distribution 

for dist in [Normal(0,1), Gamma(5, 1), Poisson(5)]
    println("The probability density function for $dist evaluated at 0 is $(pdf(dist, 0))")
end

# The `fit` function can be used to pick a reasonable method of fitting the distribution 
# (typically MLE).  MLE can be explicitly chosen using the `fit_mle` function.

y = 20 .+ randn(1000)

histogram(y, normed=true, label = "Histogram", xlim = extrema(y))

plot!(fit(Normal, y), label = "Normal Fit")
plot!(fit(Gamma, y), label = "Gamma Fit")





