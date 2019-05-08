import Pkg; Pkg.add(Pkg.PackageSpec(url="https://github.com/JuliaComputing/JuliaAcademyData.jl"))
using JuliaAcademyData; activate("Deep learning with Flux")

# <br/>
#
# # Intro to Flux.jl

#-

# In the previous course, we learned how machine learning allows us to classify data as apples or bananas with a single neuron. However, some of those details are pretty fiddly! Fortunately, Julia has a powerful package that does much of the heavy lifting for us, called [`Flux.jl`](https://fluxml.github.io/).
#
# *Using `Flux` will make classifying data and images much easier!*

#-

# ## Using `Flux.jl`
#
# We can get started with `Flux.jl` via:

## using Pkg; Pkg.add(["Flux", "Plots"])
using Flux, Plots

# #### Helpful built-in functions
#
# When working we'll `Flux`, we'll make use of built-in functionality that we've had to create for ourselves in previous notebooks.
#
# For example, the sigmoid function, σ, that we have been using already lives within `Flux`:

#nb ?σ
#jl @doc σ

#-

plot(σ, -5, 5, label="\\sigma", xlabel="x", ylabel="\\sigma\\(x\\)")

# Importantly, `Flux` allows us to *automatically create neurons* with the **`Dense`** function. For example, in the last notebook, we were looking at a neuron with 2 inputs and 1 output:
#
#  <img src="https://raw.githubusercontent.com/JuliaComputing/JuliaAcademyData.jl/master/courses/Deep%20learning%20with%20Flux/data/single-neuron.png" alt="Drawing" style="width: 500px;"/>
#
#  We could create a neuron with two inputs and one output via

model = Dense(2, 1, σ)

# This `model` object comes with places to store weights and biases:

model.W

#-

model.b

#-

typeof(model.W)

#-

x = rand(2)
model(x)

#-

σ.(model.W*x + model.b)

# Unlike in previous notebooks, note that `W` is no longer a `Vector` (1D `Array`) and `b` is no longer a number! Both are now stored in so-called `TrackedArray`s and `W` is effectively being treated as a matrix with a single row. We'll see why below.

#-

# Other helpful built-in functionality includes ways to automatically calculate gradients and also the cost function that we've used in the previous course -
#
# $$L(w, b) = \sum_i \left[y_i - f(x_i, w, b) \right]^2$$
#
# If you normalize by dividing by the total number of elements, this becomes the "mean square error" function, which in `Flux` is named **`Flux.mse`**.

methods(Flux.mse)

# ### Bringing it all together
#
# Load the datasets that contain the features of the apple and banana images.

using CSV, DataFrames

apples = DataFrame(CSV.File(datapath("data/apples.dat"), delim='\t', allowmissing=:none, normalizenames=true))
bananas = DataFrame(CSV.File(datapath("data/bananas.dat"), delim='\t', allowmissing=:none, normalizenames=true));

#-

x_apples  = [ [row.red, row.green] for row in eachrow(apples)]
x_bananas = [ [row.red, row.green] for row in eachrow(bananas)];

# Concatenate the x (features) together to create a vector of all our datapoints, and create the corresponding vector of known labels:

xs = [x_apples; x_bananas]
ys = [fill(0, size(x_apples)); fill(1, size(x_bananas))];

#-

model = Dense(2, 1, σ)

# We can evaluate the model (currently initialized with random weights) to see what the output value is for a given input:

model(xs[1])

# And of course we can examine the current loss value for that datapoint:

loss = Flux.mse(model(xs[1]), ys[1])

#-

typeof(loss)

# ### Backpropagation

model.W

#-

model.W.grad

#-

using Flux.Tracker
back!(loss)

#-

model.W.grad

# Now we have all the tools necessary to build a simple gradient descent algorithm!

#-

# ### The easy way
#
# You don't want to manually write out gradient descent algorithms every time! Flux, of course, also brings in lots of optimizers that can do this all for you.

#nb ?SGD
#jl @doc SGD

#-

#nb ?Flux.train!
#jl @doc Flux.train!

# So we can simply define our loss function, an optimizer, and then call `train!`. That's basic machine learning with Flux.jl.

model = Dense(2, 1, σ)
L(x,y) = Flux.mse(model(x), y)
opt = SGD(params(model))
Flux.train!(L, zip(xs, ys), opt)

# ## Visualize the result

contour(0:.1:1, 0:.1:1, (x, y) -> model([x,y])[].data, fill=true)
scatter!(first.(x_apples), last.(x_apples), label="apples")
scatter!(first.(x_bananas), last.(x_bananas), label="bananas")
xlabel!("mean red value")
ylabel!("mean green value")

