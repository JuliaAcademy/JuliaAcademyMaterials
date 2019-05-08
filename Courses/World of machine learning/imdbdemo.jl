# # IMDB Movie Review Sentiment Analysis Demo

#-

# ## Setup
# * Adds required packages to Julia.
# * Loads the data and a pretrained model.
# * See [07.imdb.ipynb](https://github.com/denizyuret/Knet.jl/blob/v1.1.1/tutorial/07.imdb.ipynb) to see how the model was trained.

cd("imdb")
include("imdb.jl")

# ## Sample Data
# The model was trained using 25000 movie reviews such as the following (shift-ENTER to see random example)   
# Each review was tokenized, lowercased, truncated to max 150 words and a 30,000 word vocabulary.   

r = rand(1:length(xtrn))
println(reviewstring(xtrn[r],ytrn[r]))
flush(stdout)

# ## Test
# We test the model on 25000 never before seen reviews on the test set (shift-ENTER to see random example)   
# The test accuracy is around 86%   

r = rand(1:length(xtst))
println(reviewstring(xtst[r],ytst[r]))
println(predictstring(xtst[r]))
flush(stdout)

# ## User Input
# In this cell you can enter your own review and let the model guess the sentiment

userinput = readline(stdin)
words = split(lowercase(userinput))
ex = [get!(imdbdict,wr,UNK) for wr in words]
ex[ex.>VOCABSIZE].=UNK
println(predictstring(ex))

#-



