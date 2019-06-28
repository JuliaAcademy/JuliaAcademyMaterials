# # Descending to the top
#
# The current killer app for derivatives is gradient descent — that is, the
# process of iteratively improving the parameters of an algorithm to minimize some
# measured error.  Each iteration improves on the previous set of parameters by simply
# determining which direction to "nudge" each parameter.  We could of course try changing
# each parameter individually and see which direction to move, but that's pretty tedious,
# numerically fraught, and expensive. If, however, we knew the partial derivatives with
# respect to each parameter then we could simply "descend" down the slope of our error
# function until we reach the bottom — that is, the minimum error!

# The simplest application here is simply fitting a line to some data. We'll generate
# some noisy data:

using DifferentialEquations
generate_data(ts) = solve(ODEProblem([1.0,1.0],(0.0,10.0),[1.5,1.0,3.0,1.0]) do du,u,p,t; du[1] = p[1]*u[1] - p[2]*prod(u); du[2] = -p[3]*u[2] + p[4]*prod(u); end, saveat=ts)[1,:].*(1.0.+0.02.*randn.()).+0.05.*randn.()

ts = 0:.005:1
ys = generate_data(ts)

using Plots
scatter(ts, ys)

#-

# Now we want to fit some model to this data — for a linear model we just need
# two parameters:
#
# $$
# y = m x + b
# $$

mutable struct LinearModel
    m::Float64
    b::Float64
end
(model::LinearModel)(x) = model.m*x + model.b
linear = LinearModel(randn(), randn())
plot!(ts, linear.(ts))

# Of course, we just chose our `m` and `b` at random here, of course it's not
# going to fit our data well! Let's quantify how far we are from an ideal line
# with a _loss_ function:

loss(f, xs, ys) = sum((f.(xs) .- ys).^2)

loss(linear, ts, ys)

#-

p = scatter(ts, ys, label="data")
plot!(p, ts, linear.(ts), label="model: loss $(round(Int, loss(linear, ts, ys)))")

# And now we want to try to improve the fit. To do so, we just need to make
# the loss function as small as possible. We can of course simply try a bunch
# of values and brute-force a "good" solution:

ms, bs = (-1:.01:6, -2:.05:2.5)
surface(ms, bs, [loss(LinearModel(m, b), ts, ys) for b in bs for m in ms])
xlabel!("m values")
ylabel!("b values")
title!("loss value")

# A countour plot makes it a bit more obvious where the minimum is:
contour(ms, bs, [loss(LinearModel(m, b), ts, ys) for b in bs for m in ms], levels=100)
xlabel!("m values")
ylabel!("b values")

# But that's expensive! And it becomes completely intractible as soon as we
# have more than a few parameters to fit. We can instead just try "nudging" the
# current model's values and see which way to move to improve things:

linear.m += 0.1
plot!(p, ts, linear.(ts), label="new model: loss $(round(Int, loss(linear, ts, ys)))")

# We'll either have made things better or worse — and we simply want to move
# in the direction that improves things.
#
# Essentially, what we're doing here is we're finding the derivative of our
# loss function with respect to `m`, but again, there's a bit to be desired
# here.  Instead of evaluating the loss function twice and observing the difference,
# we can just ask Julia for the derivative! This is the heart of differentiable
# programming — the ability to ask a _complete program_ for its derivative.
#
# There are several techniques that you can use to compute the derivative. We'll
# use Zygote first. It will take a second as it compiles the "adjoint" code to
# compute the derivatives. Since we're doing this in multiple dimensions, the
# two derivatives together are called the gradient:

using Zygote
grads = Zygote.gradient(linear) do m
    return loss(m, ts, ys)
end
grads

# So there we go!  Now we know the slope of our loss function — and thus
# we know which way to move to improve it! We just need to iteratively walk
# downhill!  We don't want to take too large a step, so we'll multiply each
# derivative by a "learning rate," typically called `η`.
η = 0.001
linear.m -= η*grads[1][].m
linear.b -= η*grads[1][].b

plot!(p, ts, linear.(ts), label="new model: loss $(round(Int, loss(linear, ts, ys)))")

# Now we just need to do this a bunch of times!
for i in 1:200
    grads = Zygote.gradient(linear) do m
        return loss(m, ts, ys)
    end
    linear.m -= η*grads[1][].m
    linear.b -= η*grads[1][].b
    i % 10 != 1 && continue
#nb     IJulia.clear_output(true)
    scatter(ts, ys, label="data")
    display(plot!(ts, linear.(ts), label="model (loss: $(round(loss(linear, ts, ys), digits=3)))"))
end

# That's looking pretty good now!  You might be saying this is crazy — I know
# how to do a least squares fit!  And you'd be right — in this case you can
# easily do this with a linear solve of the system of equations:
m, b = [ts ones(size(ts))] \ ys

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
(m::PolyModel)(x) = m.p[1] + m.p[2]*x + m.p[3]*x^2 #sum(m.p[i]*x^(i-1) for i in 1:length(m.p))
poly = PolyModel(rand(3))
loss(poly, ts, ys)
η = 0.001
for i in 1:1000
    grads = Zygote.gradient(poly) do m
        return loss(m, ts, ys)
    end
    poly.p .-= η.*grads[1].p
end
scatter(ts, ys)
plot!(ts, poly.(ts))
plot!(ts, PolyModel([ts.^0 ts ts.^2] \ ys).(ts))

# ## Let's see what happens when we ask for a bit more data from our "black box":

ts = 0:.04:8
ys = generate_data(ts)

scatter(ts, ys)

# Now what? This clearly won't be fit well with a low-order polynomial!
# Let's use a bit of knowledge about where this data comes from: these happen
# to be the number of rabbits in a predator-prey system! We can express the
# general behavior with a pair of differential equations:
#
# $$
# x' = \alpha x - \beta x y \\
# y' = \delta y + \gamma x y
# $$
#
# These are the classic Lotka-Volterra equations, and can be expressed in Julia
# using the DifferentialEquations package:
using DifferentialEquations
function lotka_volterra(du,u,p,t)
    x, y = u
    α, β, δ, γ = p
    du[1] = dx = α*x - β*x*y
    du[2] = dy = -δ*y + γ*x*y
end
u0 = [1.0,1.0]
tspan = extrema(ts)
p = rand(4).+1 # We don't know what this is!
## But lets see what the model looks like right now:
prob = ODEProblem(lotka_volterra,u0,tspan,p)
sol = solve(prob, Tsit5(), saveat=ts)
plot!(sol)

# So we're going to have to improve this — unlike the previous examples, we don't
# have an easy algebraic solution here! But let's try using gradient descent.
# The easiest way to get gradients out of a differential equation solver right
# now is through the DiffEqFlux package, and instead of manually converging,
# we can use the Flux package to automatically handle the gradient descent. Just
# like before, though, we compute the loss of the model evaluation — and in this
# case the model is _solving a differential equation!_
using Flux, DiffEqFlux
p = Flux.param(ones(4))
diffeq_loss(p, xs, ys) = sum(abs2, diffeq_rd(p,prob,Tsit5(),saveat=xs)[1,:] .- ys)

# This works slightly differently — we now track the gradients directly in the
# p vector:
p.grad
#-
l = diffeq_loss(p, ts, ys)
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

data = Iterators.repeated((ts, ys), 100)
opt = ADAM(0.1)
cb = function () #callback function to observe training
#nb     IJulia.clear_output(true)
    scatter(ts, ys, label="data")
    display(plot!(solve(remake(prob,p=Flux.data(p)),Tsit5(),saveat=0.1),ylim=(0,8), title="loss: $(round(Flux.data(diffeq_loss(p, ts, ys)), digits=3))"))
end
# Display the ODE with the initial parameter values.
cb()
Flux.train!((xs, ys)->diffeq_loss(p, xs, ys), [p], data, opt, cb = cb)

# # Summary
#
# You can now see the power of differentiating whole programs — we can easily
# and efficiently tune parameters without brute forcing solutions. Gradient
# descent easily extends to machine learning and artificial intelligence
# applications, but there are a variety of other places where knowing the gradient can
# be powerful and helpful.
