import Pkg; Pkg.add(Pkg.PackageSpec(url="https://github.com/JuliaComputing/JuliaAcademyData.jl"))
using JuliaAcademyData; activate("Deep learning with Flux")

# # Building a single neural network layer using `Flux.jl`
#
# In this notebook, we'll move beyond binary classification. We'll try to distinguish between three fruits now, instead of two. We'll do this using **multiple** neurons arranged in a **single layer**.

#-

# ## Read in and process data

#-

# We can start by loading the necessary packages and getting our data into working order with similar code we used at the beginning of the previous notebooks, except that now we will combine three different apple data sets, and will add in some grapes to the fruit salad!

using CSV, DataFrames, Flux, Plots

#-

## Load apple data in CSV.read for each file
apples1 = DataFrame(CSV.File(datapath("data/Apple_Golden_1.dat"), delim='\t', allowmissing=:none, normalizenames=true))
apples2 = DataFrame(CSV.File(datapath("data/Apple_Golden_2.dat"), delim='\t', allowmissing=:none, normalizenames=true))
apples3 = DataFrame(CSV.File(datapath("data/Apple_Golden_3.dat"), delim='\t', allowmissing=:none, normalizenames=true))
## And then concatenate them all together
apples = vcat(apples1, apples2, apples3)

# And now let's build an array called `x_apples` that stores data from the `red` and `blue` columns of `apples`. From `applecolnames1`, we can see that these are the 3rd and 5th columns of `apples`:

size(apples)

#-

apples[1:1, :]

#-

x_apples  = [ [apples[i, :red], apples[i, :blue]] for i in 1:size(apples, 1) ]

# Similarly, let's create arrays called `x_bananas` and `x_grapes`:

## Load data from *.dat files
bananas = DataFrame(CSV.File(datapath("data/Banana.dat"), delim='\t', allowmissing=:none, normalizenames=true))
grapes1 = DataFrame(CSV.File(datapath("data/Grape_White.dat"), delim='\t', allowmissing=:none, normalizenames=true))
grapes2 = DataFrame(CSV.File(datapath("data/Grape_White_2.dat"), delim='\t', allowmissing=:none, normalizenames=true))

## Concatenate data from two grape files together
grapes = vcat(grapes1, grapes2)

## Build x_bananas and x_grapes from bananas and grapes
x_bananas  = [ [bananas[i, :red], bananas[i, :blue]] for i in 1:size(bananas, 1) ]
x_grapes = [ [grapes[i, :red], grapes[i, :blue]] for i in 1:size(grapes, 1) ]

#-

xs = vcat(x_apples, x_bananas, x_grapes)

# ## One-hot vectors

#-

# Now we wish to classify *three* different types of fruit. It is not clear how to encode these three types using a single output variable; indeed, in general this is not possible.
#
# Instead, we have the idea of encoding $n$ output types from the classification into *vectors of length $n$*, called "one-hot vectors":
#
# $$
# \textrm{apple} = \begin{pmatrix} 1 \\ 0 \\ 0 \end{pmatrix};
# \quad
# \textrm{banana} = \begin{pmatrix} 0 \\ 1 \\ 0 \end{pmatrix};
# \quad
# \textrm{grape} = \begin{pmatrix} 0 \\ 0 \\ 1 \end{pmatrix}.
# $$
#
# The term "one-hot" refers to the fact that each vector has a single $1$, and is $0$ otherwise.
#
# Effectively, the first neuron will learn whether or not (1 or 0) the data corresponds to an apple, the second whether or not (1 or 0) it corresponds to a banana, etc.

#-

# `Flux.jl` provides an efficient representation for one-hot vectors, using advanced features of Julia so that it does not actually store these vectors, which would be a waste of memory; instead `Flux` just records in which position the non-zero element is. To us, however, it looks like all the information is being stored:

using Flux: onehot

onehot(1, 1:3)

# #### Exercise 1
#
# Make an array `labels` that gives the labels (1, 2 or 3) of each data point. Then use `onehot` to encode the information about the labels as a vector of `OneHotVector`s.

#-

# **Solution**:

labels = [ones(length(x_apples)); 2*ones(length(x_bananas)); 3*ones(length(x_grapes))]

#-

ys = [onehot(label, 1:3) for label in labels]  # onehotbatch(labels, 1:3)

# The input data is in `xs` and the one-hot vectors are in `ys`.

#-

# ## Single layer in Flux

#-

# Let's suppose that there are two pieces of input data, as in the previous single neuron notebook. Then the network has 2 inputs and 3 outputs:

include(datapath("scripts/draw_neural_net.jl"))
draw_network([2, 3])

# `Flux` allows us to express this again in a simple way:

using Flux

#-

model = Dense(2, 3, σ)

# #### Exercise 2
#
# Now what do the weights inside `model` look like? How does this compare to the diagram of the network layer above?

#-

# #### Solution

model.W

# Each of the 6 lines in the figure denotes a weight of the neuron on the right, taking as input the output of the neuron on the left. These weights are collected in the **matrix** `W`. Note that it seems to be "backwards", since it is designed to multiply vectors of length 2 (the input size):

x = rand(2)
model.W * x

# The whole `model` object represents a set of three sigmoidal neurons. Again, we can use `model` as a function and apply it to input data to get the model's prediction from that data:

model(x)

# This is equivalent to the following calculation:

σ.( (model.W) * x .+ (model.b) )

# Note that here we have again used Julia's **broadcasting** capability, in which the function $\sigma$ is applied to each element of the vector `W * x` in turn. This elementwise application of the function is implicit (i.e. not written out) in most of the literature on machine learning, but it is very much clearer to make this explicit, as Julia forces us to do.
#
# [Recall that we also use `.` to access the fields `W` and `b` in the `model` object; this is a different use of `.`.]

#-

# ## Training the model

#-

# Despite the fact that the model is now more complicated than the single neuron from the previous notebook, the beauty of `Flux.jl` is that the rest of the training process **looks exactly the same**!

#-

# #### Exercise 3
#
# Implement training for this model.

#-

# **Solution**:

loss(x, y) = Flux.mse(model(x), y)

#-

data = zip(xs, ys);

#-

opt = SGD(params(model), 0.01)
## give a list of the parameters that will be modified

#-

for i in 1:100
    Flux.train!(loss, data, opt)
end

#-

model.W

#-

model.b

# #### Exercise 4
#
# Visualize the result of the learning for each neuron. Since each neuron is sigmoidal, we can get a good idea of the function by just plotting a single contour level where the function takes the value 0.5, using the `contour` function with keyword argument `levels=[0.5, 0.501]`.

using Plots
plot()

contour!(0:0.01:1, 0:0.01:1, (x,y)->model([x,y]).data[1], levels=[0.5, 0.501], color = cgrad([:blue, :blue]))
contour!(0:0.01:1, 0:0.01:1, (x,y)->model([x,y]).data[2], levels=[0.5,0.501], color = cgrad([:green, :green]))
contour!(0:0.01:1, 0:0.01:1, (x,y)->model([x,y]).data[3], levels=[0.5,0.501], color = cgrad([:red, :red]))

scatter!(first.(x_apples), last.(x_apples), m=:cross, label="apples")
scatter!(first.(x_bananas), last.(x_bananas), m=:circle, label="bananas")
scatter!(first.(x_grapes), last.(x_grapes), m=:square, label="grapes")

# #### Exercise 5
#
# Interpret the results by checking which fruit each neuron was supposed to learn and what it managed to achieve.

#-

# #### Solution

#-

# We see that two of the hyperplanes have been learnt correctly, the one that separates bananas from the rest, and the one that separates grapes from the rest. However, the hyperplane for the neuron that was supposed to learn to distinguish apples is not at all correct.
#
# It is hopefully clear why this is so: there *is no way* to separate apples from non-apples with a *single* hyperplane, given this data.
#
# The problem is that this network is so simple that it is *not capable of learning a good representation of the data*: the class of functions that are modelled by this single layer, with only 9 parameters, is not complex enough.
#
# In order to learn how to classify three different types of fruit, we must thus move to more complex neural networks, and towards **deep learning**!

