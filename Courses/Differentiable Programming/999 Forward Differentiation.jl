module ADPackages #src
# Some settings for running this interactively #src
interactive_use = false #src
plotting_off    = false #src
using BenchmarkTools #src
if interactive_use #src
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 0.1 #src
end #src

# # Using AD packages
#
# While a theoretical understanding behind how automatic differentiation work
# is interesting, in practice, being able to use existing AD packages might
# be good enough so this is where we will start.
# Julia has many excellent packages for AD. The purpose of this lecture
# for you to try a few of them out and apply them to real problems like solving
# optimization problems using AD.

# ## ForwardDiff
#
# The most popular package for AD in Julia is [`ForwardDiff`](https://github.com/JuliaDiff/ForwardDiff.jl).
# It uses something called *forward mode automatic differentiation* using "Dual numbers"
# and operator overloading. Forward mode AD is efficient if the function
# being differentiated is $\mathbb{R}^N -> \mathbb{R}^M$ with $N <= M$. In other words,
# the dimension of the input argument should be equal or greater than the number of
# output arguments.
# Foroward mode AD is however not so efficient for example when `M = 1` and `N` is big, which is common in
# e.g. machine learning. In these cases *reverse mode automatic differentiation* is
# more efficient, and we will look at that later.

# ### Derivative of scalar functions
#
# To have a function to start experimenting with, we define the (scalar to
# scalar) function `f(x)` and its analytical derivative
# <a href="https://www.wolframalpha.com/input/?i=simplify+derivative+exp(x)+%2F+(sin(x)%5E3+%2B+cos(x)%5E3)">`fp(x)`</a>

f(x) = exp(x) / (sin(x)^3 + cos(x)^3);
fp(x) = exp(x) * (3sin(x) + sin(3x) + 2cos(3x)) / 2(sin(x)^3 + cos(x)^3)^2;

# We can visualize the function and the derivative (here using the Plots.jl package)
if !plotting_off #src
using Plots
end #src
xs = 0.0:0.01:1.4
if !plotting_off #src
plot(xs, [f.(xs), fp.(xs)]; labels = ["f" "fp"])
end #src

# The only thing to do to differentiate this function using AD is to import `ForwardDiff`
# and call the `ForwardDiff.derivative` function with the function we want to differentiate and
# at what point we want the derivative. The function should take a scalar as an argument
# and return a scalar and that is what our function `f` above does.

import ForwardDiff
ad_derivative = ForwardDiff.derivative(f, pi/4)
analytical_derivative = fp(pi/4)
@assert ad_derivative ≈ analytical_derivative #src
@show ad_derivative - analytical_derivative;

# We can see that the AD derivative is "exact" (at least exact in the sense of floating
# point precision) to the analytical ones.

# #### Performance
#
# It can be interesting to see what the performance of using AD vs the the performance of the
# analytical derivative. We can use the benchmark framework [`BenchmarkTools.jl`](https://github.com/JuliaCI/BenchmarkTools.jl)
# to help us make accurate benchmarks even for functions that execute very quickly.

using BenchmarkTools

@assert VERSION < v"1.1" "revise benchmarking for 1.1" #src

############################################################ #src
# TODO: THIS IS ONLY TRUE ON MASTER, THE OPTIMIZATION DOES NOT HAPPEN ON JULIA 1.0.1                      #src
# If we do some napkin math, we can see that the reported time here is completely bogus!                  #src
# Making the not too terrible assumption that a CPU can do one operation per clockcycle and that  #src
# the CPU you are using has approximately 3 GHz then one instruction should take ~0.3 ns which is  #src
# much longer than the reported time here. And computing this derivative need to compute `sin`, `cos`  #src
# etc which take many CPU instructions.  #src
# We have now encounted one of hard problems when benchmarking, to make sure that the computer is actually  #src
# computing the things you want it to. What happens here is that the Julia compiler is able  #src
# to figure out that the input we gave the function (`pi/4`) is a constant and the optimizer  #src
# then figures out the result during compile time. We can see this as follows:  #src
# #src
#g() = ForwardDiff.derivative(f, pi/4) #src
# #src
## Looking at the generated code for #src
# #src
#@code_llvm g() #src
# #src
# We can see that the function has been optimized to just return a single value. #src
# The exact value looks a bit odd because it is written #src
# but reinterpreting it as a Float64 value we can see that it is just the same #src
# derivative as was returned from ForwardDiff: #src
# #src
#@show reinterpret(Float64, 0x4008D06AE62ADC94) #src
# #src
#@show ForwardDiff.derivative(f, pi/4) #src
# #src
# One way we can trick the optimizer (at least with the current version of it) to not optimized #src
# away the computation #src
# is to encapsulate our value in a container like a [`Ref`](https://docs.julialang.org/en/v1/base/c/#Core.Ref) #src
# #src
##################################################################################### #src

# A function can be benchmarked by prepending the call with `@btime`.
# This will run the function multiple time and collect statistics on the
# time it took to execute the function:

println("Original function")
@btime f(pi/4);
println("AD derivative")
@btime ForwardDiff.derivative(f, pi/4);
println("Analytical derivative")
@btime fp(pi/4);

# We can here see that for the way we implemented the analytical derivative there is virtually no
# performance difference between the AD version and the analyitical version.
# However, it should be noted that the current version of the analytical function is not the
# one with the best performance. Feel free to try to improve it.

# A nice thing about AD is that you don't have to spend time making sure that
# not only is the derivative correct, but that it is also efficiently computed.
# By using AD, you get "free" performance for the derivative by optimizing the function itself.

# #### Second derivatives
#
# For scalar functions `ForwardDiff` does not come with built in functionality for computing
# the second derivative. It is, fortunately, very easy to create such a function ourselves.
# The second derivative is just the derivative of the derivative which we can implement as:

derivative2(f, x) = ForwardDiff.derivative(z -> ForwardDiff.derivative(f, z), x)

@assert derivative2(f, pi/4) ≈  -6.203532787672101 #src
@show derivative2(f, pi/4);

# Here, we created an anonymous function that computes derivatives of `f`,
# `z -> ForwardDiff.derivative(f, z)`, and then we used that as the input
# derivative for another `ForwardDiff.derivative` call. The derivative
# of the derivative gives the second derivative. Simple!
# Feel free to compute the second derivative by hand and verify that the result
# is correct (and appreciate that AD savees you from doing it in the first place).
# The correct analytical result is `-2sqrt(2) * exp(pi/4)`.

# #### Differentiating functions with parameters
#
# It is common to have a function that depends on some parameters that are considered
# fixed in the context of differentiation.
# An example of this might be the function `g` (and its derivative `gp) below:

g(x, a) = (x - a)^3
gp(x, a) = 3*(x - a)^2

# Here, `g` has the parameter `a` and we want to take the derivative with respect to `x`.
# Recall that `ForwardDiff.derivative` needs a function that takes only one
# argument, but `g` above takes two arguments, so it clearly cannot be used directly.
# The solution is to create what is typically called a "closure", which is a new function
# that "closes" over some parameter space.
# For example, consider:

const a = 3
g2(x) = g(x, a)

@show (2 - a)^3
@show g2(2);

# We now have a new function `g2` which takes only one argument (`x`) and "closes" over the
# variable `a`. We can now differentiate `g2`.

@show ForwardDiff.derivative(g2, 2.0)
@show gp(2.0, 3);

# It is possible to write this a bit more succintly using an anonymous function

ForwardDiff.derivative(x -> g(x, a), 2.0)

# ## Gradients
#
# If our function takes a vector as input and returns a scalar, the derivative
# is a vector (called a gradient) which gives the sensitivity of the output with respect to
# all inputs. A quite common function to use in examples involving optimization
# is the [Rosenbrock function](https://en.wikipedia.org/wiki/Rosenbrock_function), defined as:

function rosenbrock(x)
   a = one(eltype(x))
   b = 100 * a
   result = zero(eltype(x))
   for i in 1:length(x)-1
       result += (a - x[i])^2 + b*(x[i+1] - x[i]^2)^2
   end
   return result
end

# Evaluating `rosenbrock` with a vector input indeed gives a scalar back.

rosenbrock([1.0, 0.5, 2.0])
@assert rosenbrock([1.0, 0.5, 2.0]) == 331.5 #src

# We can see how this function looks by plotting it

xs = -3:0.1:3
ys = -3:0.1:3
zs = [rosenbrock([x, y]) for x in xs, y in ys]
if !plotting_off #src
contour(xs, ys, zs; levels=40)
end #src

# Evaluating the `gradient` is almost as simple as the `derivative`:

x_vec = [1.0, 2.0]
ForwardDiff.gradient(rosenbrock, x_vec)

# ### Performance
#
# When evaluating the gradient we are dealing with `Vector`s which require allocation
# so there are a few extra performance considerations for `gradient` compared to `derivative`

# #### Preallocating output
#
# `ForwardDiff.gradient` returns a `Vector` which needs to be allocated. We can preallocate
# this so `ForwardDiff` doesn't need to to it by itself.

z_vec = similar(x_vec)
ForwardDiff.gradient!(z_vec, rosenbrock, x_vec)
@show z_vec;

# The result was now stored in the, by us, allocated `z_vec`. We can check the performance difference

println("No preallocation")
@btime ForwardDiff.gradient(rosenbrock, $x_vec)
println("No preallocation")
@btime ForwardDiff.gradient!($z_vec,rosenbrock, $x_vec)

# We can see that we do one allocation less and that performance is significantly
# improved.

# #### Preallocating internal datastructures
#
# Even though we preallocated the output vector there are some
# internal data structures used by ForwardDiff that we in addition can preallocate
# This is done by creating a `GradientConfig`:

gradient_cache = ForwardDiff.GradientConfig(rosenbrock, x_vec)
@btime ForwardDiff.gradient!($z_vec, rosenbrock, $x_vec, $gradient_cache);

# Which as we can see further increase performance.
# All of this is documented in the ForwardDiff manual so the takehome message is
# to read the manual of the package one uses. There are often valuable information
# there that could (like in this case) significantly improve performance.

# ### Solving optimization using ForwardDiff
#
# [`Optim`](https://github.com/JuliaNLSolvers/Optim.jl) is a package that has many useful optimization routines.
# If you can provide the gradient (and even sec) you get access to optimization routines that can
# have significantly better performance. Lets use first try to just optimize the function without
# using any gradient information. This will default to using the
# [Nelder-Mead method](https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method).
# We also stick a `@btime` in there to see how long time the optimization routine take.
using Random; Random.seed!(1234) #src
using Optim
x0 = zeros(10)
@btime optimize(rosenbrock, x0)

# We can see that we require approximately 620 evaluations of `rosenbrock` before
# a minimum was found. Let's try giving `Optim` the gradient as well:
#
using Random; Random.seed!(1234) #src
const gradient_cache_optim = ForwardDiff.GradientConfig(rosenbrock, x0)
const rosenbrock_gradient! = (z, x) -> ForwardDiff.gradient!(z,rosenbrock, x, gradient_cache_optim)

@btime optimize(rosenbrock, rosenbrock_gradient!, x0)

# Now we only called the function 92 times (and in addition the gradient 153 times)
# Looking at the time taken this was almost a 3x speedup. The minimum we found had a
# was also significantly smaller than when we used Nelder-Mead

# ## Hessians
# It is also possible to compute Hessians (second derivative) in a very similar way to `gradient`

ForwardDiff.hessian(rosenbrock, [1.0, 2.0])

# We leave out the details for `ForwardDiff.hessian` here and instead refer to the [`ForwardDiff.hessian`](http://www.juliadiff.org/ForwardDiff.jl/stable/user/api.html#Hessians-of-f(x::AbstractArray)::Real-1)
# documentation.

# ## Jacobians
#
# If our function takes a vector as argument and returns a vector, the derivative is a `Matrix` which
# is known as the Jacobian. This derivative is useful when we want to solve a nonlinear system of
# equations.
# Let's consider the following function that slightly resembles the `rosenbrock` function:

function rosenbrock_vector(x)
    return [
        1 - x[1],
        10(x[2]-x[1]^2)
    ]
end

# Indeed calling this function with a vector argument returns a vector:

rosenbrock_vector([1.0, 0.5])

# By now, we should be quite familiar with the ForwarDiff interface and you might
# even guess how we should compute the Jacobian:

ForwardDiff.jacobian(rosenbrock_vector, [1.0, 0.5])

# The Jacobian functionality could be used for example in [`NLsolve.jl`](https://github.com/JuliaNLSolvers/NLsolve.jl),
# which is a nonlinear equation solver.

# ### Summary `ForwardDiff`
# This should get you started with using `ForwardDiff`. If your functions do not have too big vectors as
# input arguments, the performance should be good. It will likely not beat a carefully tuned
# analytical implementaion of the derivative but it is oftan that from a productivity point of view
# it is worth using AD.

# ## Revisediff
#
# ForwardDiff is using what is known as forward mode differentiation.
# If the number of input parameters is large and the output is just a scalar
# then reverse mode differentation is likely more effective.
#
# TODO..

end #src
