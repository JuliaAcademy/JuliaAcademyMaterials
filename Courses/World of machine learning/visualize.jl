import Pkg; Pkg.add(Pkg.PackageSpec(url="https://github.com/JuliaComputing/JuliaAcademyData.jl"))
using JuliaAcademyData; activate("World of machine learning")

# # Visual Q&A Demo

#-

# ## Setup
# 1-Installs required packages   
# 2-Downloads sample data and a pretrained model.     

cd(datapath("visualize"))
include("demosetup.jl") 
include("src/main.jl")

# ## Initialization
# 1-Loads the sample demo data (image features,questions,vocabulary).   
#     * w2i : words to one-hot index   : w2i["cube"] = 30  
#     * a2i : answer to one-hot index  : a2i["gray"] = 8
#     * i2w : one-hot index to words   : i2w[2] = "Are"   
#     * i2a : one-hot index to answers : i2a[5] = "large"   
# 2-Loads the pretrained model, and its hyper-parameters `o`.     

feats,qstsns,(w2i,a2i,i2w,i2a) = loadDemoData("data/demo/");
_,Mrun,o = loadmodel("models/macnet.jld2";onlywrun=true);
global atype = typeof(params(Mrun)[1].value) <: Array ? Array{Float32} : KnetArray{Float32}

# ## Sample Data
# 1-Randomly selects (question,image) pair from the sample data   
# 2-Make predictions for the question and checks whether the prediction is correct   

rnd        = rand(1:length(qstsns))
inst       = qstsns[rnd]
feat       = atype(feats[:,:,:,rnd:rnd])
question   = Array{Int}(inst[2])
answer     = inst[3];
family     = inst[4];
results,prediction,interoutputs = singlerun(Mrun,feat,question;p=o[:p],selfattn=o[:selfattn],gating=o[:gating]);
interoutputs = first.(interoutputs)
answer==prediction[1]

#-

(i2a[interoutputs],prediction[1])

#-

img = load("data/demo/CLEVR_v1.0/images/val/$(inst[1])")

#-

textq  = i2w[question];
println("Question: ",join(textq," "))
texta  = i2a[answer];
println("Answer: $(texta)\nPrediction: $(i2a[prediction]) ")

# ## User Data
# You can enter your own question about the image and test whether the prediction is correct

userinput = readline(stdin)
words = split(userinput) # tokenize(userinput)
question = [get!(w2i,wr,1) for wr in words]
results,prediction = singlerun(Mrun,feat,question;p=o[:p],selfattn=o[:selfattn],gating=o[:gating]);
println("Question: $(join(i2w[question]," "))")
println("Prediction: $(i2a[prediction])")

# ## Visualize
# `visualize` function visualizes attention maps for each time step of the mac network

visualize(img,results;p=o[:p])

#-



