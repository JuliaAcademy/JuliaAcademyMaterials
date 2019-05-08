# This is a copy of Knet/data/imdb.jl and Knet/tutorial/07.imdb.ipynb cleaned up

# Based on https://github.com/fchollet/keras/raw/master/keras/datasets/imdb.py
# Also see https://github.com/fchollet/keras/raw/master/examples/imdb_lstm.py
# Also see https://github.com/ilkarman/DeepLearningFrameworks/raw/master/common/utils.py

using Pkg; for p in ("PyCall","JSON","JLD2"); haskey(Pkg.installed(),p) || Pkg.add(p); end
using PyCall,JSON,JLD2,Random
@pyimport numpy as np

"""

    imdb()

Load the IMDB Movie reviews sentiment classification dataset from
https://keras.io/datasets and return (xtrn,ytrn,xtst,ytst,dict) tuple.

# Keyword Arguments:
- url=https://s3.amazonaws.com/text-datasets: where to download the data (imdb.npz) from.
- dir=Pkg.dir("Knet/data"): where to cache the data.
- maxval=nothing: max number of token values to include. Words are ranked by how often they occur (in the training set) and only the most frequent words are kept. nothing means keep all, equivalent to maxval = vocabSize + pad + stoken.
- maxlen=nothing: truncate sequences after this length. nothing means do not truncate.
- seed=0: random seed for sample shuffling. Use system seed if 0.
- pad=true: whether to pad short sequences (padding is done at the beginning of sequences). pad_token = maxval.
- stoken=true: whether to add a start token to the beginning of each sequence. start_token = maxval - pad.
- oov=true: whether to replace words >= oov_token with oov_token (the alternative is to skip them). oov_token = maxval - pad - stoken.

"""
function imdb(;
              url = "https://s3.amazonaws.com/text-datasets",
              dir = "./", # joinpath(@__DIR__, "imdb"),
              data="imdb.npz",
              dict="imdb_word_index.json",
              jld2="imdb.jld2",
              maxval=nothing,
              maxlen=nothing,
              seed=0, oov=true, stoken=true, pad=true
              )
    global _imdb_xtrn,_imdb_ytrn,_imdb_xtst,_imdb_ytst,_imdb_dict
    if !(@isdefined _imdb_xtrn)
        isdir(dir) || mkpath(dir)
        jld2path = joinpath(dir,jld2)
        if !isfile(jld2path)
            @info("Downloading IMDB...")
            datapath = joinpath(dir,data)
            dictpath = joinpath(dir,dict)
            isfile(datapath) || download("$url/$data",datapath)
            isfile(dictpath) || download("$url/$dict",dictpath)
            d = np.load(datapath)
            _imdb_xtrn = map(a->np.asarray(a,dtype=np.int32), get(d, "x_train"))
            _imdb_ytrn = Array{Int8}(get(d, "y_train") .+ 1)
            _imdb_xtst = map(a->np.asarray(a,dtype=np.int32), get(d, "x_test"))
            _imdb_ytst = Array{Int8}(get(d, "y_test") .+ 1)
            _imdb_dict = Dict{String,Int32}(JSON.parsefile(dictpath))
            JLD2.@save jld2path _imdb_xtrn _imdb_ytrn _imdb_xtst _imdb_ytst _imdb_dict
            #rm(datapath)
            #rm(dictpath)
        end
        @info("Loading IMDB...")
        JLD2.@load jld2path _imdb_xtrn _imdb_ytrn _imdb_xtst _imdb_ytst _imdb_dict
    end
    if seed != 0; Random.seed!(seed); end
    xs = [_imdb_xtrn;_imdb_xtst]
    if maxlen == nothing; maxlen = maximum(map(length,xs)); end
    if maxval == nothing; maxval = maximum(map(maximum,xs)) + pad + stoken; end
    if pad; pad_token = maxval; maxval -= 1; end
    if stoken; start_token = maxval; maxval -= 1; end
    if oov; oov_token = maxval; end
    function _imdb_helper(x,y)
        rp = randperm(length(x))
        newy = y[rp]
        newx = similar(x)
        for i in 1:length(x)
            xi = x[rp[i]]
            if oov
                xi = map(w->(w<=oov_token ? w : oov_token), xi)
            else
                xi = filter(w->(w<=oov_token), xi)
            end
            if stoken
                xi = [ start_token; xi ]
            end
            if length(xi) > maxlen
                xi = xi[end-maxlen+1:end]
            end
            if pad && length(xi) < maxlen
                xi = append!(repeat([pad_token], maxlen-length(xi)), xi)
            end
            newx[i] = xi
        end
        newx,newy
    end
    xtrn,ytrn = _imdb_helper(_imdb_xtrn,_imdb_ytrn)
    xtst,ytst = _imdb_helper(_imdb_xtst,_imdb_ytst)
    return xtrn,ytrn,xtst,ytst,_imdb_dict
end

### From 07.imdb.ipynb

using Pkg
for p in ("Knet",) #"ProgressMeter")
    haskey(Pkg.installed(),p) || Pkg.add(p)
end

EPOCHS=3          # Number of training epochs
BATCHSIZE=64      # Number of instances in a minibatch
EMBEDSIZE=125     # Word embedding size
NUMHIDDEN=100     # Hidden layer size
MAXLEN=150        # maximum size of the word sequence, pad shorter sequences, truncate longer ones
VOCABSIZE=30000   # maximum vocabulary size, keep the most frequent 30K, map the rest to UNK token
NUMCLASS=2        # number of output classes
DROPOUT=0.0       # Dropout rate
LR=0.001          # Learning rate
BETA_1=0.9        # Adam optimization parameter
BETA_2=0.999      # Adam optimization parameter
EPS=1e-08         # Adam optimization parameter

using Knet # : Knet
#ENV["COLUMNS"]=92                     # column width for array printing
#include(Knet.dir("data","imdb.jl"))   # defines imdb loader

#@doc imdb

#@time 
(xtrn,ytrn,xtst,ytst,imdbdict)=imdb(maxlen=MAXLEN,maxval=VOCABSIZE);

#summary.((xtrn,ytrn,xtst,ytst,imdbdict))

# Words are encoded with integers
#rand(xtrn)'

# Each word sequence is padded or truncated to length 150
#length.(xtrn)'

# Define a function that can print the actual words:
imdbvocab = Array{String}(undef,length(imdbdict))
for (k,v) in imdbdict; imdbvocab[v]=k; end
imdbvocab[VOCABSIZE-2:VOCABSIZE] = ["<unk>","<s>","<pad>"]
function reviewstring(x,y=0)
    x = x[x.!=VOCABSIZE] # remove pads
    """$(("Sample","Negative","Positive")[y+1]) review:\n$(join(imdbvocab[x]," "))"""
end

# Hit Ctrl-Enter to see random reviews:
#r = rand(1:length(xtrn))
#println(reviewstring(xtrn[r],ytrn[r]))

# Here are the labels: 1=negative, 2=positive
#ytrn'

using Knet: param, dropout, RNN

struct SequenceClassifier; input; rnn; output; end

SequenceClassifier(input::Int, embed::Int, hidden::Int, output::Int) =
    SequenceClassifier(param(embed,input), RNN(embed,hidden,rnnType=:gru), param(output,hidden))

function (sc::SequenceClassifier)(input; pdrop=0)
    embed = sc.input[:, permutedims(hcat(input...))]
    embed = dropout(embed,pdrop)
    hidden = sc.rnn(embed)
    hidden = dropout(hidden,pdrop)
    return sc.output * hidden[:,:,end]
end

using Knet: minibatch
dtrn = minibatch(xtrn,ytrn,BATCHSIZE;shuffle=true)
dtst = minibatch(xtst,ytst,BATCHSIZE)
#length.((dtrn,dtst))

# For running experiments
#using Knet: train!, Adam
#import ProgressMeter

function trainresults(file,model)
    # if (print("Train from scratch? ");readline()[1]=='y')
    #     updates = 0; prog = ProgressMeter.Progress(EPOCHS * length(dtrn))
    #     function callback(J)
    #         ProgressMeter.update!(prog, updates)
    #         return (updates += 1) <= prog.n
    #     end
    #     opt = Adam(lr=LR, beta1=BETA_1, beta2=BETA_2, eps=EPS)
    #     train!(model, dtrn; callback=callback, optimizer=opt, pdrop=DROPOUT)
    #     Knet.gc()
    #     Knet.save(file,"model",model)
    # else
        isfile(file) || download("http://people.csail.mit.edu/deniz/models/tutorial/$file",file)
        model = Knet.load(file,"model")
    # end
    return model
end

#using Knet: nll, accuracy
model = SequenceClassifier(VOCABSIZE,EMBEDSIZE,NUMHIDDEN,NUMCLASS)
#nll(model,dtrn), nll(model,dtst), accuracy(model,dtrn), accuracy(model,dtst)

model = trainresults("imdbmodel.jld2",model);

# 33s (0.059155148f0, 0.3877507f0, 0.9846153846153847, 0.8583733974358975)
#nll(model,dtrn), nll(model,dtst), accuracy(model,dtrn), accuracy(model,dtst)

predictstring(x)="\nPrediction: " * ("Negative","Positive")[argmax(Array(vec(model([x]))))]
UNK = VOCABSIZE-2
str2ids(s::String)=[(i=get(imdbdict,w,UNK); i>=UNK ? UNK : i) for w in split(lowercase(s))]

# Here we can see predictions for random reviews from the test set; hit Ctrl-Enter to sample:
# r = rand(1:length(xtst))
# println(reviewstring(xtst[r],ytst[r]))
# println(predictstring(xtst[r]))

# Here the user can enter their own reviews and classify them:
# println(predictstring(str2ids(readline(stdin))))
