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
# ### Univariate Data
#
# First we will look at how to estimate/plot univariate distributions of data. 
# As we saw in the intro module, we can plot histograms of continuous datafrom the 
# **`StatsBase`** package.

using StatsBase, Plots

y = randn(1000)

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

p = plot(kde(y), label = "Bandwidth: Auto", w=2)
for bw in .1:.2:.9
    plot!(p, kde(y; bandwidth = bw), label = "Bandwidth: $bw", w=2)
end
histogram!(p, y, alpha = .1, normed=true)


# #### Averaged Shifted Histograms (ASH)









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





