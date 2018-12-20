# ## Going deep: Deep neural networks
# 
# So far, we've learned that if we want to classify more than two fruits, we'll need to go beyond using a single neuron and use *multiple* neurons to get multiple outputs. We can think of stacking these multiple neurons together in a single neural layer.
# 
# Even so, we found that using a single neural layer was not enough to fully distinguish between bananas, grapes, **and** apples. To do this properly, we'll need to add more complexity to our model. We need not just a neural network, but a *deep neural network*. 
# 
# There is one step remaining to build a deep neural network. We have been saying that a neural network takes in data and then spits out `0` or `1` predictions that together declare what kind of fruit the picture is. However, what if we instead put the output of one neural network layer into another neural network layer?
# 
# This gets pictured like this below:
# 
# <img src="data/deep-neural-net.png" alt="Drawing" style="width: 500px;"/>
# 
# On the left we have 3 data points in blue. Those 3 data points each get fed into 4 neurons in purple. Each of those 4 neurons produces a single output, but those output are each fed into three neurons (the second layer of purple). Each of those 3 neurons spits out a single value, and those values are fed as inputs into the last layer of 6 neurons. The 6 values that those final neurons produce are the output of the neural network. This is a deep neural network.

#-

# ### Why would a deep neural network be better?
# 
# This is a little perplexing when you first see it. We used neurons to train the model before: why would sticking the output from neurons into other neurons help us fit the data better? The answer can be understood by drawing pictures. Geometrically, the matrix multiplication inside of a layer of neurons is streching and rotating the axis that we can vary:
# 
# ![](data/17-raw.png)
# 
# A nonlinear transformation, such as the sigmoid function, then adds a bump to this linearly-transformed data:
# 
# ![](data/17-linear1.png)
# 
# Resulting in a bit more of the data accounted for:
# 
# ![](data/17-nonlinear1.png)
# 
# Now let's repeat this process. When we send the data through another layer of neurons, we get another rotation and another bump:
# 
# ![](data/17-linear2.png)
# 
# ![](data/17-nonlinear2.png)
# 
# 
# Visually, we see that if we keep doing this process we can make the axis line up with any data. What this means is that **if we have enough layers, then our neural network can approximate any model**. 
# 
# The trade-off is that with more layers we have more parameters, so it may be harder (i.e. computationally intensive) to train the neural network. But we have the guarantee that the model has enough freedom such that there are parameters that will give the correct output. 
# 
# Because this model is so flexible, the problem is reduced to that of learning: do the same gradient descent method on this much larger model (but more efficiently!) and we can make it classify our data correctly. This is the power of deep learning.

#-

# #### Solution

## Simple cartoon plots to demonstrate pseudo-rotation and "bump"
using Plots
xs = range(0, stop=10, length=200)
ys = randn.() .+ xs .+ 0.5.*xs.^1.5 .+ 3.0.*sin.(clamp.(xs .- 5, 0, Inf)) .+ 2.6.*clamp.(.-abs.(xs .- 2), -2, 0)
scatter(xs, ys, label="", ticks = false)
savefig("data/17-raw.png")

fit1 = [xs ones(size(xs))] \ ys
linear1(x) = fit1[1]*x + fit1[2]
plot!(linear1, label="")

scatter(xs, ys .- linear1.(xs), ticks = false, label = "")
savefig("data/17-linear1.png")

nonlinearity(x) = clamp.(2.0.*x .- 2, -2, 2)
plot!(nonlinearity, label = "")

ys2 = ys .- linear1.(xs) .- nonlinearity.(xs)
scatter(xs, ys2, ticks = false, label = "")
savefig("data/17-nonlinear1.png")

fit2 = [xs ones(size(xs))] \ ys2
linear2(x) = fit2[1]*x + fit2[2]
plot!(linear2)

ys3 = ys2 .- linear2.(xs)
scatter(xs .- 2, .-ys3, ticks = false, label = "")
savefig("data/17-linear2.png")

plot!(nonlinearity, label = "")

ys4 = .-ys3 .- nonlinearity.(xs .- 2)
scatter(xs, ys4, ticks = false, label = "")
savefig("data/17-nonlinear2.png")

