import Pkg; Pkg.add(Pkg.PackageSpec(url="https://github.com/JuliaComputing/JuliaAcademyData.jl"))
using JuliaAcademyData; activate("Deep learning with Flux")

# <br /><br /><br />
# # Recognizing handwritten digits using a neural network

#-

# We have now reached the point where we can tackle a very interesting task: applying the knowledge we have gained with machine learning in general, and `Flux.jl` in particular, to create a neural network that can recognize handwritten digits! The data are from a data set called MNIST, which has become a classic in the machine learning world.
#
# [We could also try to apply the techniques to the original images of fruit instead. However, the fruit images are much larger than the MNIST images, which makes the learning a suitable neural network too slow.]

#-

# ## Data munging

#-

# As we know, the first difficulty with any new data set is locating it, understanding what format it is stored in, reading it in and decoding it into a useful data structure in Julia.
#
# The original MNIST data is available [here](http://yann.lecun.com/exdb/mnist); see also the [Wikipedia page](https://en.wikipedia.org/wiki/MNIST_database). However, the format that the data is stored in is rather obscure.
#
# Fortunately, various packages in Julia provide nicer interfaces to access it. We will use the one provided by `Flux.jl`.
#
# The data are images of handwritten digits, and the corresponding labels that were determined by hand (i.e. by humans). Our job will be to get the computer to **learn** to recognize digits by learning, as usual, the function that relates the input and output data.

#-

# ### Loading and examining the data

#-

# First we load the required packages:

using Flux, Flux.Data.MNIST, Images

# Now we read in the data:

labels = MNIST.labels();
images = MNIST.images();

# We see that `images` is a `Vector`, i.e. an `Array{T, 1}` with a complicated parameter `T`. It has length

length(labels)

# But we can just look at the first handful to get a sense of the contents

images[1:5]

#-

labels[1:5]' # transposed to match the above

# So the $i$th entry of the array is the data for the $i$th image.

typeof(images[1])

# As with the fruit images from the start of the course, the image is an array of color blocks, except that now each pixel just has a grey scale.
#
# To see the actual numeric content of the image, we can do, for example

Float64.(images[1])

# ### Setting up a neural network

#-

# In the previous notebooks, we arranged the input data for Flux as a `Vector` of `Vector`s.
# Now we will use an alternative arrangement, as a matrix, since that allows `Flux` to use matrix operations, which are more efficient.
#
# The column $i$ of the matrix is a vector consisting of the $i$th data point $\mathbf{x}^{(i)}$.  Similarly, the desired outputs are given as a matrix, with the $i$th column being the desired output $\mathbf{y}^{(i)}$.

n_inputs = unique(length.(images))[]

#-

n_outputs = length(unique(labels))

# #### Creating the features
#
# We want to create a vector of feature vectors, each with the floating point values of the 784 pixels.
#
# An image is a matrix of colours, but now we need a vector of floating point numbers instead. To do so, we just arrange all of the elements of the matrix in a certain way into a single list; fortunately, Julia already provides the function `vec` to do so!

#-

# Let's use a subset of $N=5,000$ of the total $60,000$ images available in order to hold out test images that our model hasn't been trained on.

preprocess(img) = vec(Float64.(img))

#-

xs = preprocess.(images[1:5000]);

# #### Creating the labels
#
# We can just use `Flux.onehot` to create them:

ys = [Flux.onehot(label, 0:9) for label in labels[1:5000]];

# #### Create the batched matrices for efficiency

#-

# Let's also create a function so we can easily create independent batches from arbitrary segments of the original dataset.

function create_batch(r)
    xs = [preprocess(img) for img in images[r]]
    ys = [Flux.onehot(label, 0:9) for label in labels[r]]
    return (Flux.batch(xs), Flux.batch(ys))
end

# We'll train our model on the first 5000 images.

trainbatch = create_batch(1:5000);

# ## Setting up the neural network

#-

# Now we must set up a neural network. Since the data is complicated, we may expect to need several layers.
# But we can start with a single layer.
#
# - The network will take as inputs the vectors $\mathbf{x}^{(i)}$, so the input layer has $n$ nodes.
#
# - The output will be a one-hot vector encoding the digit from 1 to 9 or 0 that is desired. There are 10 possible categories, so we need an output layer of size 10.
#
# It is then our task as neural network designers to insert layers between these input and output layers, whose weights will be tuned during the learning process. *This is an art, not a science*! But major advances have come from finding a good structure for the network.

model = Chain(Dense(n_inputs, n_outputs, identity), softmax)
L(x,y) = Flux.crossentropy(model(x), y)
opt = SGD(params(model))
@time Flux.train!(L, [trainbatch], opt)
@time Flux.train!(L, [trainbatch], opt)

# ## Training

#-

# Just as before, use `repeated` to create an **iterator**. It does not copy the data 100 times, which would be very wasteful; it just gives an object that repeatedly loops over the same data:

Iterators.repeated(trainbatch, 100);

# We can see what the total current loss is (after training just a handful to times above):

L(trainbatch...)

#-

@time Flux.train!(L, [trainbatch], opt)

#-

L(trainbatch...)

# ### Using callbacks

#-

# The `train!` function can take an optional keyword argument, `cb` (short for "**c**all**b**ack"). A callback function is a function that you provide as an argument to a function `f`, which "calls back" your function every so often.
#
# This provides the possibility to provide a function that is called at each step or every so often during the training process.
# A common use case is to provide a visual trace of the training process by printing out the current value of the `loss` function:

callback() = @show(L(trainbatch...))

Flux.train!(L, Iterators.repeated(trainbatch, 3), opt; cb = callback)

# However, it is expensive to calculate the complete `loss` function and it is not necessary to output it every step. So `Flux` also provides a function `throttle`, that provides a mechanism to call a given function at most once every certain number of seconds:

Flux.train!(L, Iterators.repeated(trainbatch, 40), opt; cb = Flux.throttle(callback, 1))

# Of course, that's just measuring the loss over the same data it's training on. It'd be more representative to test against novel data. In fact, let's track the performance of both as we continue training our model. In order to do so, we need to create a batch of test data.

testbatch = create_batch(5001:10000);

#-

using Printf
train_loss = Float64[]
test_loss = Float64[]
function update_loss!()
    push!(train_loss, L(trainbatch...).data)
    push!(test_loss, L(testbatch...).data)
    @printf("train loss = %.2f, test loss = %.2f\n", train_loss[end], test_loss[end])
end

#-

Flux.train!(L, Iterators.repeated(trainbatch, 1000), opt;
                cb = Flux.throttle(update_loss!, 1))

#-

using Plots
plot(1:length(train_loss), train_loss, xlabel="~seconds of training", ylabel="loss", label="train")
plot!(1:length(test_loss), test_loss, label="test")

# ## Testing phase

#-

# We now have trained a model, i.e. we have found the parameters `W` and `b` for the network layer(s). In order to **test** if the learning procedure was really successful, we check how well the resulting trained network performs when we test it with images that the network has not yet seen!
#
# Often, a dataset is split up into "training data" and "test (or validation) data" for this purpose, and indeed the MNIST data set has a separate pool of training data. We can instead use the images that we have not included in our reduced training process.

i = 5001
display(images[i])
labels[i], findmax(model(preprocess(images[i]))) .- (0, 1)

#-

model(preprocess(images[i]))

# ## Evaluation
#
# What percent of images are we correctly classifying if we take the highest element to be the chosen answer?

prediction(i) = findmax(model(preprocess(images[i])))[2]-1 # returns (max_value, index)

#-

sum(prediction(i) == labels[i] for i in 1:5000)/5000

#-

sum(prediction(i) == labels[i] for i in 5001:10000)/5000

# ## Improving the prediction

#-

# So far we have used a single layer. In order to improve the prediction, we probably need to use more layers. Try adding more layers yourself and see how the performance changes.

n_hidden = 20
model = Chain(Dense(n_inputs, n_hidden, relu),
              Dense(n_hidden, n_outputs, identity), softmax)
L(x,y) = Flux.crossentropy(model(x), y)
opt = ADAM(params(model))

#-

train_loss = Float64[]
test_loss = Float64[]
Flux.train!(L, Iterators.repeated(trainbatch, 1000), opt;
            cb = Flux.throttle(update_loss!, 1))

#-

plot(1:length(train_loss), train_loss, xlabel="~seconds of training", ylabel="loss", label="train")
plot!(1:length(test_loss), test_loss, label="test")

# ## What about image structure?

#-

# As a final note, notice that our model doesn't take into account any aspect of the image's connected-ness.

using Random
p = randperm(28)
images[1][p,p]

#-

show(preprocess(images[1]))

#-

#nb ?Conv
#jl @doc Conv

