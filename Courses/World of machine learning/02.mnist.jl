import Pkg; Pkg.add(Pkg.PackageSpec(url="https://github.com/JuliaComputing/JuliaAcademyData.jl"))
using JuliaAcademyData; activate("World of machine learning")
cd(datapath("mnist"))

# # Load and minibatch MNIST data
# (c) Deniz Yuret, 2018

#-

# * Objective: Learning the structure of the MNIST dataset and usage of the Knet.Data struct.
# * Prerequisites: [The iteration interface](https://docs.julialang.org/en/v1/manual/interfaces/#man-interface-iteration-1)
# * Knet: mnistdata, mnistview (used and explained)
# * Knet: dir, minibatch (used by mnist.jl)

#-

## This loads MNIST handwritten digit recognition dataset.
## Traininig and test data go to variables dtrn and dtst
using Knet: Knet, minibatch
include(Knet.dir("data","mnist.jl"))  # defines mnistdata and mnistview
dtrn,dtst = mnistdata(xtype=Array{Float32});

#-

## dtrn contains 600 minibatches of 100 images (total 60000)
## dtst contains 100 minibatches of 100 images (total 10000)
length.((dtrn,dtst))

#-

## Each minibatch is an (x,y) pair where x is 100 28x28x1 images and y contains their labels.
## Here is the first minibatch in the test set:
(x,y) = first(dtst)
summary.((x,y))

#-

## Here is the first five images from the test set:
using Images, ImageMagick
hcat([mnistview(x,i) for i=1:5]...)

#-

## Here are their labels (0x0a=10 is used to represent 0)
@show y[1:5];

#-

## dtrn and dtst are implemented as Julia iterators (see https://docs.julialang.org/en/v1/manual/interfaces)
## This means they can be used in for loops, i.e. `for (x,y) in dtst`
cnt = zeros(Int,10)
for (x,y) in dtst
    for label in y
        cnt[label] += 1
    end
end
@show cnt;

# # Convolutional Neural Networks
# (c) Deniz Yuret, 2018

#-

# * Objectives: See the effect of sparse and shared weights implemented by convolutional networks.
# * Prerequisites: MLP models (04.mlp.ipynb), KnetArray, param, param0, dropout, relu, nll
# * Knet: conv4, pool, mat (explained)
# * Knet: dir, gpu, minibatch, KnetArray (used by mnist.jl)
# * Knet: SGD, train!, Train, load, save (used by trainresults)

# ## Introduction to convolution

## Convolution operator in Knet
using Knet: conv4
@doc conv4

#-

## Convolution in 1-D
@show w = reshape([1.0,2.0,3.0], (3,1,1,1))
@show x = reshape([1.0:7.0...], (7,1,1,1))
@show y = conv4(w, x);  # size Y = X - W + 1 = 5 by default

#-

## Padding
@show y2 = conv4(w, x, padding=(1,0));  # size Y = X + 2P - W + 1 = 7 with padding=1
## To preserve input size (Y=X) for a given W, what padding P should we use?

#-

## Stride
@show y3 = conv4(w, x; padding=(1,0), stride=3);  # size Y = 1 + floor((X+2P-W)/S)

#-

## Mode
@show y4 = conv4(w, x, mode=0);  # Default mode (convolution) inverts w
@show y5 = conv4(w, x, mode=1);  # mode=1 (cross-correlation) does not invert w

#-

## Convolution in more dimensions
x = reshape([1.0:9.0...], (3,3,1,1))

#-

w = reshape([1.0:4.0...], (2,2,1,1))

#-

y = conv4(w, x)

#-

## Convolution with multiple channels, filters, and instances
## size X = [X1,X2,...,Xd,Cx,N] where d is the number of dimensions, Cx is channels, N is instances
x = reshape([1.0:18.0...], (3,3,2,1)) 

#-

## size W = [W1,W2,...,Wd,Cx,Cy] where d is the number of dimensions, Cx is input channels, Cy is output channels
w = reshape([1.0:24.0...], (2,2,2,3));

#-

## size Y = [Y1,Y2,...,Yd,Cy,N]  where Yi = 1 + floor((Xi+2Pi-Wi)/Si), Cy is channels, N is instances
y = conv4(w,x)

# See http://cs231n.github.io/assets/conv-demo/index.html for an animated example.

#-

# ## Introduction to Pooling

## Pooling operator in Knet
using Knet: pool
@doc pool

#-

## 1-D pooling example
@show x = reshape([1.0:6.0...], (6,1,1,1))
@show pool(x);

#-

## Window size
@show pool(x; window=3);  # size Y = floor(X/W)

#-

## Padding
@show pool(x; padding=(1,0));  # size Y = floor((X+2P)/W)

#-

## Stride
@show x = reshape([1.0:10.0...], (10,1,1,1));
@show pool(x; stride=4);  # size Y = 1 + floor((X+2P-W)/S)

#-

## Mode (using Array here; not all modes are implemented on the CPU)
using Knet: KnetArray
x = Array(reshape([1.0:6.0...], (6,1,1,1)))
@show x
@show pool(x; padding=(1,0), mode=0)  # max pooling
@show pool(x; padding=(1,0), mode=1)  # avg pooling
## @show pool(x; padding=(1,0), mode=2); # avg pooling excluding padded values (is not implemented on CPU so will error)

#-

## More dimensions
x = reshape([1.0:16.0...], (4,4,1,1))

#-

pool(x)

#-

## Multiple channels and instances
x = reshape([1.0:32.0...], (4,4,2,1))

#-

## each channel and each instance is pooled separately
pool(x)  # size Y = (Y1,...,Yd,Cx,N) where Yi are spatial dims, Cx and N are identical to input X

# ## Experiment setup

## Load data (see 02.mnist.ipynb)
using Knet: Knet, KnetArray, gpu, minibatch
include(Knet.dir("data","mnist.jl"))  # Load data
dtrn,dtst = mnistdata();              # dtrn and dtst = [ (x1,y1), (x2,y2), ... ] where xi,yi are minibatches of 100

#-

(x,y) = first(dtst)
summary.((x,y))

#-

## For running experiments
using Knet: SGD, train!, nll, zeroone
import ProgressMeter
    
function trainresults(file,model; o...)
    if (print("Train from scratch? ");readline()[1]=='y')
        results = Float64[]; updates = 0; prog = ProgressMeter.Progress(60000)
        function callback(J)
            if updates % 600 == 0
                push!(results, nll(model,dtrn), nll(model,dtst), zeroone(model,dtrn), zeroone(model,dtst))
                ProgressMeter.update!(prog, updates)
            end
            return (updates += 1) <= 60000
        end
        train!(model, dtrn; callback=callback, optimizer=SGD(lr=0.1), o...)
        Knet.save(file,"results",reshape(results, (4,:)))
    end
    isfile(file) || download("http://people.csail.mit.edu/deniz/models/tutorial/$file",file)
    results = Knet.load(file,"results")
    println(minimum(results,dims=2))
    return results
end

# ## A convolutional neural network model for MNIST

## Redefine Linear layer (See 03.lin.ipynb):
using Knet: param, param0
struct Linear; w; b; end
(f::Linear)(x) = (f.w * mat(x) .+ f.b)
mat(x)=reshape(x,:,size(x)[end])  # Reshapes 4-D tensor to 2-D matrix so we can use matmul
Linear(inputsize::Int,outputsize::Int) = Linear(param(outputsize,inputsize),param0(outputsize))

#-

## Define a convolutional layer:
struct Conv; w; b; end
(f::Conv)(x) = pool(conv4(f.w,x) .+ f.b)
Conv(w1,w2,cx,cy) = Conv(param(w1,w2,cx,cy), param0(1,1,cy,1))

#-

## Define a convolutional neural network:
struct CNN; layers; end

#-

## Weight initialization for a multi-layer convolutional neural network
## h[i] is an integer for a fully connected layer, a triple of integers for convolution filters and tensor inputs
## use CNN(x,h1,h2,...,hn,y) for a n hidden layer model
function CNN(h...)  
    w = Any[]
    x = h[1]
    for i=2:length(h)
        if isa(h[i],Tuple)
            (x1,x2,cx) = x
            (w1,w2,cy) = h[i]
            push!(w, Conv(w1,w2,cx,cy))
            x = ((x1-w1+1)÷2,(x2-w2+1)÷2,cy) # assuming conv4 with p=0, s=1 and pool with p=0,w=s=2
        elseif isa(h[i],Integer)
            push!(w, Linear(prod(x),h[i]))
            x = h[i]
        else
            error("Unknown layer type: $(h[i])")
        end
    end
    CNN(w)
end;

#-

using Knet: dropout, relu
function (m::CNN)(x; pdrop=0)
    for (i,layer) in enumerate(m.layers)
        p = (i <= length(pdrop) ? pdrop[i] : pdrop[end])
        x = dropout(x, p)
        x = layer(x)
        x = (layer == m.layers[end] ? x : relu.(x))
    end
    return x
end

#-

lenet = CNN((28,28,1), (5,5,20), (5,5,50), 500, 10)
summary.(l.w for l in lenet.layers)

#-

using Knet: nll
(x,y) = first(dtst)
nll(lenet,x,y)

# ## CNN vs MLP

using Plots; default(fmt=:png,ls=:auto)
ENV["COLUMNS"] = 92

#-

@time cnn = trainresults("cnn.jld2", lenet; pdrop=(0,0,.3)); # 406s [8.83583e-5, 0.017289, 0.0, 0.0048]
## Note that training will take a very long time without a GPU

#-

isfile("mlp.jld2") || download("http://people.csail.mit.edu/deniz/models/tutorial/mlp.jld2","mlp.jld2")
mlp = Knet.load("mlp.jld2","results");

#-

## Comparison to MLP shows faster convergence, better generalization
plot([mlp[1,:], mlp[2,:], cnn[1,:], cnn[2,:]],ylim=(0.0,0.1),
     labels=[:trnMLP :tstMLP :trnCNN :tstCNN],xlabel="Epochs",ylabel="Loss")  

#-

plot([mlp[3,:], mlp[4,:], cnn[3,:], cnn[4,:]],ylim=(0.0,0.03),
    labels=[:trnMLP :tstMLP :trnCNN :tstCNN],xlabel="Epochs",ylabel="Error")  

# ## Convolution vs Matrix Multiplication

## Convolution and matrix multiplication can be implemented in terms of each other.
## Convolutional networks have no additional representational power, only statistical efficiency.
## Our original 1-D example
@show w = reshape([1.0,2.0,3.0], (3,1,1,1))
@show x = reshape([1.0:7.0...], (7,1,1,1))
@show y = conv4(w, x);  # size Y = X - W + 1 = 5 by default

#-

## Convolution as matrix multiplication (1)
## Turn w into a (Y,X) sparse matrix
w2 = Float64[3 2 1 0 0 0 0; 0 3 2 1 0 0 0; 0 0 3 2 1 0 0; 0 0 0 3 2 1 0; 0 0 0 0 3 2 1]

#-

@show y2 = w2 * mat(x);

#-

## Convolution as matrix multiplication (2)
## Turn x into a (W,Y) dense matrix (aka the im2col operation)
## This is used to speed up convolution with known efficient matmul algorithms
x3 = Float64[1 2 3 4 5; 2 3 4 5 6; 3 4 5 6 7]

#-

@show w3 = [3.0 2.0 1.0]
@show y3 = w3 * x3;

#-

## Matrix multiplication as convolution
## This could be used to make a fully connected network accept variable sized inputs.
w = reshape([1.0:6.0...], (2,3))

#-

x = reshape([1.0:3.0...], (3,1))

#-

y = w * x

#-

## Consider w with size (Y,X)
## Treat each of the Y rows of w as a convolution filter
w2 = copy(reshape(Array(w)', (3,1,1,2)))

#-

## Reshape x for convolution
x2 = reshape(x, (3,1,1,1))

#-

## Use conv4 for matrix multiplication
y2 = conv4(w2, x2; mode=1)

#-

## So there is no difference between the class of functions representable with an MLP vs CNN.
## Sparse connections and weight sharing give CNNs more generalization power with images.
## Number of parameters in MLP256: (256x784)+256+(10x256)+10 = 203530
## Number of parameters in LeNet: (5*5*1*20)+20+(5*5*20*50)+50+(500*800)+500+(10*500)+10 = 431080

