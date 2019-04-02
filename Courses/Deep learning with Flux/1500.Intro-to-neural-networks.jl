# <br /><br />
#
# ## Neural networks
#
# Now that we know what neurons are, we are ready for the final step: the neural network!. A neural network is literally made out of a network of neurons that are connected together.
#
# So far, we have just looked at single neurons, that only have a single output.
# What if we want multiple outputs?
#
#
# ### Multiple output models
#
# What if we wanted to distinguish between apples, bananas, *and* grapes? We could use *vectors* of `0` or `1` values to symbolize each output.
#
# <img src="data/fruit-salad.png" alt="Drawing" style="width: 300px;"/>
#
# The idea of using vectors is that different directions in the space of outputs encode information about different types of inputs.

#-

# Now we extend our previous model to give multiple outputs by repeating it with different weights. For the first element of the array we'd use:
#
# $$\sigma(x;w^{(1)},b^{(1)}) := \frac{1}{1 + \exp(-w^{(1)} \cdot x + b^{(1)})};$$
#
# then for the second we'd use
#
# $$\sigma(x;w^{(2)},b^{(2)}) := \frac{1}{1 + \exp(-w^{(2)} \cdot x + b^{(2)})};$$
#
# and if you wanted $n$ outputs, you'd have for each one
#
# $$\sigma(x;w^{(i)},b^{(i)}) := \frac{1}{1 + \exp(-w^{(i)} \cdot x + b^{(i)})}.$$

#-

# Notice that these equations are all the same, except for the parameters, so we can write this model more succinctly, as follows. Let's write $b$ in an array:
#
# $$b=\left[\begin{array}{c}
# b_{1}\\
# b_{2}\\
# \vdots\\
# b_{n}
# \end{array}\right]$$
#
# and put our array of weights as a matrix:
#
# $$ \mathsf{W}=\left[\begin{array}{c}
# \\
# \\
# \\
# \\
# \end{array}\begin{array}{cccc}
# w_{1}^{(1)} & w_{2}^{(1)} & \ldots & w_{n}^{(1)}\\
# w_{1}^{(2)} & w_{2}^{(2)} & \ldots & w_{n}^{(2)}\\
# \vdots & \vdots &  & \vdots\\
# w_{1}^{(n)} & w_{2}^{(n)} & \ldots & w_{n}^{(n)}
# \end{array}\right]
# $$
#
# We can write this all in one line as:
#
# $$\sigma(x;w,b)= \left[\begin{array}{c}
# \sigma^{(1)}\\
# \sigma^{(2)}\\
# \vdots\\
# \sigma^{(n)}
# \end{array}\right] = \frac{1}{1 + \exp(-\mathsf{W} x + b)}$$
#
# $\mathsf{W} x$ is the operation called "matrix multiplication"
#
# [Show small matrix multiplication]

W = [10 1;
     20 2;
     30 3]
x = [3;
     2]
W*x

# It takes each column of weights and does the dot product against $x$ (remember, that's how $\sigma^{(i)}$ was defined) and spits out a vector from doing that with each column. The result is a vector, which makes this version of the function give a vector of outputs which we can use to encode larger set of choices.
#
# Matrix multiplication is also interesting since **GPUs (Graphics Processing Units, i.e. graphics cards) are basically just matrix multiplication machines**, which means that by writing the equation this way, the result can be calculated really fast.

#-

# This "multiple input and multiple output" version of the sigmoid function is known as a *layer of neurons*.
#
# Previously we worked with a single neuron, which we visualized as
#
# <img src="data/single-neuron.png" alt="Drawing" style="width: 300px;"/>
#
# where we have two pieces of data (green) coming into a single neuron (pink) that returned a single output. We could use this single output to do binary classification - to identify an image of a fruit as `1`, meaning banana or as `0`, meaning not a banana (or an apple).
#
# To do non-binary classification, we can use a layer of neurons, which we can visualize as
#
# <img src="data/single-layer.png" alt="Drawing" style="width: 300px;"/>
#
# We now have stacked a bunch of neurons on top of each other to hopefully work together and train to output results of more complicated features.
#
# We still have two input pieces of data, but now have several neurons, each of which produces an output for a given binary classification:
# * neuron 1: "is it an apple?"
# * neuron 2: "is it a banana?"
# * neuron 3: "is it a grape?"

#-

# # Multiple outputs with Flux.jl

#-

# First step: load the data.

using CSV, DataFrames, Flux, Plots
## Load apple data in CSV.read for each file
apples1 = DataFrame(CSV.File("data/Apple_Golden_1.dat", delim='\t', allowmissing=:none, normalizenames=true))
apples2 = DataFrame(CSV.File("data/Apple_Golden_2.dat", delim='\t', allowmissing=:none, normalizenames=true))
apples3 = DataFrame(CSV.File("data/Apple_Golden_3.dat", delim='\t', allowmissing=:none, normalizenames=true))
## And then concatenate them all together
apples = vcat(apples1, apples2, apples3)
bananas = DataFrame(CSV.File("data/Banana.dat", delim='\t', allowmissing=:none, normalizenames=true))
grapes1 = DataFrame(CSV.File("data/Grape_White.dat", delim='\t', allowmissing=:none, normalizenames=true))
grapes2 = DataFrame(CSV.File("data/Grape_White_2.dat", delim='\t', allowmissing=:none, normalizenames=true))
grapes = vcat(grapes1, grapes2)

#-

## Extract out the features and construct the corresponding labels
x_apples  = [ [apples[i, :red], apples[i, :blue]] for i in 1:size(apples, 1) ]
x_bananas  = [ [bananas[i, :red], bananas[i, :blue]] for i in 1:size(bananas, 1) ]
x_grapes = [ [grapes[i, :red], grapes[i, :blue]] for i in 1:size(grapes, 1) ]
xs = vcat(x_apples, x_bananas, x_grapes)
ys = vcat(fill([1,0,0], size(x_apples)),
          fill([0,1,0], size(x_bananas)),
          fill([0,0,1], size(x_grapes)))

# ### One-hot vectors

#-

# Recall:
#
# <img src="data/fruit-salad.png" alt="Drawing" style="width: 300px;"/>

#-

# `Flux.jl` provides an efficient representation for one-hot vectors, using advanced features of Julia so that it does not actually store these vectors, which would be a waste of memory; instead `Flux` just records in which position the non-zero element is. To us, however, it looks like all the information is being stored:

using Flux: onehot

onehot(2, 1:3)

#-

ys = vcat(fill(onehot(1, 1:3), size(x_apples)),
          fill(onehot(2, 1:3), size(x_bananas)),
          fill(onehot(3, 1:3), size(x_grapes)))

# ## The core algorithm from the previous lecture

## model = Dense(2, 1, σ)
## L(x,y) = Flux.mse(model(x), y)
## opt = SGD(params(model))
## Flux.train!(L, zip(xs, ys), opt)

# ### Visualization

using Plots
plot()

contour!(0:0.01:1, 0:0.01:1, (x,y)->model([x,y]).data[1], levels=[0.5, 0.51], color = cgrad([:blue, :blue]))
contour!(0:0.01:1, 0:0.01:1, (x,y)->model([x,y]).data[2], levels=[0.5,0.51], color = cgrad([:green, :green]))
contour!(0:0.01:1, 0:0.01:1, (x,y)->model([x,y]).data[3], levels=[0.5,0.51], color = cgrad([:red, :red]))

scatter!(first.(x_apples), last.(x_apples), m=:cross, label="apples", color = :blue)
scatter!(first.(x_bananas), last.(x_bananas), m=:circle, label="bananas", color = :green)
scatter!(first.(x_grapes), last.(x_grapes), m=:square, label="grapes", color = :red)

