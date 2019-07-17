#nb import Pkg; Pkg.add(Pkg.PackageSpec(url="https://github.com/JuliaComputing/JuliaAcademyData.jl"))
#nb using JuliaAcademyData; activate("Differentiable Programming")
datapath(p) = joinpath("../../../JuliaAcademyData.jl/courses/Differentiable Programming", p) #src

# # Descending to the top
#
# One killer app for derivatives is gradient descent. This is the process of
# incrementally improving some algorithm by adjusting its "knobs" (the tuneable
# parameters) based on its performance for some existing data.
#
# Each step incrementally improves on the previous set of parameters by
# determining which way to "nudge" each parameter in order to improve on its output.
# The trick is finding this direction efficiently.
#
# We could of course try changing each parameter individually and see which direction to move, but that's pretty tedious,
# numerically fraught, and expensive. If, however, we knew the _partial derivatives_ with
# respect to each parameter then we could simply "descend" down the slope of our error
# function until we reach the bottom — that is, the minimum error!
#
# We all know how to draw a line on a graph — it just requires knowing the slope
# and intercept of the equation
#
# $$
# y = m x + b
# $$
#
# What we want to do, though, is the inverse problem.  We have a dataset and we
# want to find a line that best fits it.  We can use gradient descent to do this:
#
# <img src="https://raw.githubusercontent.com/JuliaComputing/JuliaAcademyData.jl/master/courses/Differentiable%20Programming/images/020-linear.gif" alt="Fitting a line" />
#
# This is trivial — and there are better ways to do this in the first place —
# but the beauty of gradient descent is that it extends to much more complicated
# examples. For example, we can even fit a differential equation with this method:
#
# <img src="https://raw.githubusercontent.com/JuliaComputing/JuliaAcademyData.jl/master/courses/Differentiable%20Programming/images/020-diffeq.gif" alt="Fitting a differential equation" />
#
# But let's examine that line-fitting first as its simplicity makes a number of
# points clear. To begin, let's load some data.

using DifferentialEquations, CSV #src
import Random #src
Random.seed!(20) #src
generate_data(ts) = solve(ODEProblem([1.0,1.0],(0.0,10.0),[1.5,1.0,3.0,1.0]) do du,u,p,t; du[1] = p[1]*u[1] - p[2]*prod(u); du[2] = -p[3]*u[2] + p[4]*prod(u); end, saveat=ts)[1,:].*(1.0.+0.02.*randn.()).+0.05.*randn.() #src
ts = 0:.005:1 #src
ys = generate_data(ts) #src
CSV.write(datapath("data/020-descent-data.csv"), (t=ts, y=ys)) #src

using CSV
df = CSV.read(datapath("data/020-descent-data.csv"))
#-
using Plots
scatter(df.t, df.y, xlabel="time", label="data")

# Now we want to fit some model to this data — for a linear model we just need
# two parameters:
#
# $$
# y = m x + b
# $$
#
# Lets create a structure that represents this:

mutable struct LinearModel
    m::Float64
    b::Float64
end
(model::LinearModel)(x) = model.m*x + model.b

# And create a randomly picked model to see how we do:
linear = LinearModel(randn(), randn())
plot!(df.t, linear.(df.t), label="model")

# Of course, we just chose our `m` and `b` at random here, of course it's not
# going to fit our data well! Let's quantify how far we are from an ideal line
# with a _loss_ function:

loss(f, xs, ys) = sum((f.(xs) .- ys).^2)
loss(linear, df.t, df.y)

# That's a pretty big number — and we want to decrease it. Let's update our plot
# to include this loss value in the legend so we can keep track:

p = scatter(df.t, df.y, xlabel="time", label="data")
plot!(p, df.t, linear.(df.t), label="model: loss $(round(Int, loss(linear, df.t, df.y)))")

# And now we want to try to improve the fit. To do so, we just need to make
# the loss function as small as possible. We can of course simply try a bunch
# of values and brute-force a "good" solution. Plotting this as a three-dimensional
# surface gives us the imagery of a hill, and our goal is to find our way to the bottom:

ms, bs = (-1:.01:6, -2:.05:2.5)
surface(ms, bs, [loss(LinearModel(m, b), df.t, df.y) for b in bs for m in ms], xlabel="m values", ylabel="b values", title="loss value")

# A countour plot makes it a bit more obvious where the minimum is:

contour(ms, bs, [loss(LinearModel(m, b), df.t, df.y) for b in bs for m in ms], levels=150, xlabel="m values", ylabel="b values")

# But building those graphs are expensive! And it becomes completely intractible as soon as we
# have more than a few parameters to fit. We can instead just try nudging the
# current model's values to simply figure out which direction is "downhill":

linear.m += 0.1
plot!(p, df.t, linear.(df.t), label="new model: loss $(round(Int, loss(linear, df.t, df.y)))")

# We'll either have made things better or worse — but either way it's easy to
# see which way to change `m` in order to improve our model.
#
# Essentially, what we're doing here is we're finding the derivative of our
# loss function with respect to `m`, but again, there's a bit to be desired
# here.  Instead of evaluating the loss function twice and observing the difference,
# we can just ask Julia for the derivative! This is the heart of differentiable
# programming — the ability to ask a _complete program_ for its derivative.
#
# There are several techniques that you can use to compute the derivative. We'll
# use the [Zygote package](https://fluxml.ai/Zygote.jl/latest/) first. It will
# take a second as it compiles the "adjoint" code to compute the derivatives.
# Since we're doing this in multiple dimensions, the two derivatives together
# are called the gradient:

using Zygote
grads = Zygote.gradient(linear) do m
    return loss(m, df.t, df.y)
end
grads

# So there we go! Zygote saw that the model was gave it — `linear` — had two fields and thus
# it computed the derivative of the loss function with respect to those two
# parameters.
#
# Now we know the slope of our loss function — and thus
# we know which way to move each parameter to improve it! We just need to iteratively walk
# downhill!  We don't want to take too large a step, so we'll multiply each
# derivative by a "learning rate," typically called `η`.
η = 0.001
linear.m -= η*grads[1][].m
linear.b -= η*grads[1][].b

plot!(p, df.t, linear.(df.t), label="updated model: loss $(round(Int, loss(linear, df.t, df.y)))")

# Now we just need to do this a bunch of times!

for i in 1:200
    grads = Zygote.gradient(linear) do m
        return loss(m, df.t, df.y)
    end
    linear.m -= η*grads[1][].m
    linear.b -= η*grads[1][].b
    i > 40 && i % 10 != 1 && continue
#nb     IJulia.clear_output(true)
    scatter(df.t, df.y, label="data", xlabel="time", legend=:topleft)
    display(plot!(df.t, linear.(df.t), label="model (loss: $(round(loss(linear, df.t, df.y), digits=3)))"))
end

#src === CREATING GIF ===
f = let #src
    linear = LinearModel(randn(), randn()) #src
    f = @gif for i in 1:150 #src
        grads = Zygote.gradient(linear) do m #src
            return loss(m, df.t, df.y) #src
        end #src
        linear.m -= η*grads[1][].m #src
        linear.b -= η*grads[1][].b #src
        scatter(df.t, df.y, label="data", xlabel="time", legend=:topleft) #src
        plot!(df.t, linear.(df.t), label="model (loss: $(round(loss(linear, df.t, df.y), digits=3)))") #src
    end #src
    mv(f.filename, datapath("images/020-linear.gif")) #src
end #src



# That's looking pretty good now!  You might be saying this is crazy — I know
# how to do a least squares fit!  And you'd be right — in this case you can
# easily do this with a linear solve of the system of equations:
m, b = [df.t ones(size(df.t))] \ df.y

@show (m, b)
@show (linear.m, linear.b);

# ## Exercise: Use gradient descent to fit a quadratic model to the data
#
# Obviously a linear fit leaves a bit to be desired here. Try to use the same
# framework to fit a quadratic model.  Check your answer against the algebraic
# solution with the `\` operator.

struct PolyModel
    p::Vector{Float64}
end
function (m::PolyModel)(x)
    r = m.p[1]*x^0
    for i in 2:length(m.p)
        r += m.p[i]*x^(i-1)
    end
    return r
end
poly = PolyModel(rand(3))
loss(poly, df.t, df.y)
η = 0.001
for i in 1:1000
    grads = Zygote.gradient(poly) do m
        return loss(m, df.t, df.y)
    end
    poly.p .-= η.*grads[1].p
end
scatter(df.t, df.y, label="data", legend=:topleft)
plot!(df.t, poly.(df.t), label="model by descent")
plot!(df.t, PolyModel([df.t.^0 df.t df.t.^2] \ df.y).(df.t), label="model by linear solve")

# ## Nonlinearities
#
# Let's see what happens when load a bit more data:

let #src
    ts = 0:.04:8 #src
    ys = generate_data(ts) #src
    CSV.write(datapath("data/020-descent-data-2.csv"), (t=ts, y=ys)) #src
end #src
df2 = CSV.read(datapath("data/020-descent-data-2.csv"))
scatter(df2.t, df2.y, xlabel="t", label="more data")

# Now what? This clearly won't be fit well with a low-order polynomial, and
# even a high-order polynomial will get things wrong once we try to see what
# will happen in the "future"!
#
# Let's use a bit of knowledge about where this data comes from: these happen
# to be the number of rabbits in a predator-prey system! We can express the
# general behavior with a pair of differential equations:
#
# $$
# x' = \alpha x - \beta x y \\
# y' = -\delta y + \gamma x y
# $$
#
# * $x$ is the number of rabbits (🐰)
# * $y$ is the number of wolves (🐺)
# * $\alpha$ and $\beta$ describe the growth and death rates for the rabbits
# * $\gamma$ and $\delta$ describe the growth and death rates for the wolves
#
# But we don't know what those rate constants are — all we have is the population
# of rabbits over time.
#
# These are the classic Lotka-Volterra equations, and can be expressed in Julia
# using the [DifferentialEquations](https://github.com/JuliaDiffEq/DifferentialEquations.jl) package:
using DifferentialEquations
function rabbit_wolf(du,u,p,t)
    🐰, 🐺 = u
    α, β, δ, γ = p
    du[1] = α*🐰 - β*🐰*🐺
    du[2] = -δ*🐺 + γ*🐰*🐺
end
u0 = [1.0,1.0]
tspan = extrema(df2.t)
p = rand(4).+1 # We don't know what this is!
## But lets see what the model looks like right now:
prob = ODEProblem(rabbit_wolf,u0,tspan,p)
sol = solve(prob, Tsit5(), saveat=df2.t)

scatter(df2.t, df2.y, xlabel="t", label="more data")
plot!(sol, label=["rabbits","wolves"])

# So we're going to have to improve this — unlike the previous examples, we don't
# have an easy algebraic solution here! But let's try using gradient descent.
# The easiest way to get gradients out of a differential equation solver right
# now is through the [DiffEqFlux package](https://github.com/JuliaDiffEq/DiffEqFlux.jl), and instead of manually converging,
# we can use the Flux package to automatically (and more smartly) handle the gradient descent. Just
# like before, though, we compute the loss of the model evaluation — and in this
# case the model is _solving a differential equation!_
using Flux, DiffEqFlux
p = Flux.param(ones(4))
diffeq_loss(p, xs, ys) = sum(abs2, diffeq_rd(p,prob,Tsit5(),saveat=df2.t)[1,:] .- df2.y)

# This works slightly differently — we now track the gradients directly in the
# p vector:
p.grad
#-
l = diffeq_loss(p, df2.t, df2.y)
#-
DiffEqFlux.Tracker.back!(l) # we need to back-propagate our tracking of the gradients
p.grad # but now we can see the gradients involved in that computation!

# So now we can do exactly the same thing as before: iteratively update the parameters
# to descend to a (hopefully) well-fit model:
#
# ```julia
# p.data .-= η*p.grad
# ```
#
# But we can be a bit smarter about this just by asking Flux to handle everything
# with its train! function and associated functionality:

data = Iterators.repeated((df2.t, df2.y), 150)
opt = ADAM(0.1)
history = Any[] #src
cb = function () #callback function to observe training
#nb     IJulia.clear_output(true)
    plt = scatter(df2.t, df2.y, label="data", ylabel="population (thousands)", xlabel="t")
    display(plot!(plt, solve(remake(prob,p=Flux.data(p)),Tsit5(),saveat=0.1),
        ylim=(0,8), label=["rabbits","wolves"], title="loss: $(round(Flux.data(diffeq_loss(p, df2.t, df2.y)), digits=3))"))
    push!(history, plt) #src
end
Flux.train!((xs, ys)->diffeq_loss(p, xs, ys), [p], data, opt, cb = cb)

f = @gif for i in 1:length(history) #src
    plot(history[i]) #src
end #src
mv(f.filename, datapath("images/020-diffeq.gif")) #src

# # Summary
#
# You can now see the power of differentiating whole programs — we can easily
# and efficiently tune parameters without brute forcing solutions. Gradient
# descent easily extends to machine learning and artificial intelligence
# applications — and there are a number of tricks that can increase its efficiency
# and help avoid local minima. There are a variety of other places where knowing the gradient can
# be powerful and helpful.
