using Pkg; for p in ("Knet",); haskey(Pkg.installed(),p) || Pkg.add(p); end

using Knet

struct CharLM; input; rnn; output; end

CharLM(vocab::Int,input::Int,hidden::Int; o...) =
    CharLM(Embed(vocab,input), RNN(input,hidden; o...), Linear(hidden,vocab))

function (c::CharLM)(x; pdrop=0, hidden=nothing)
    x = c.input(x)                # (B,T)->(X,B,T)
    x = dropout(x, pdrop)
    x = c.rnn(x, hidden=hidden)   # (H,B,T)
    x = dropout(x, pdrop)
    x = reshape(x, size(x,1), :)  # (H,B*T)
    return c.output(x)            # (V,B*T)
end

struct Embed; w; end

Embed(vocab::Int,embed::Int)=Embed(param(embed,vocab))

(e::Embed)(x) = e.w[:,x]

struct Linear; w; b; end

Linear(input::Int, output::Int)=Linear(param(output,input), param0(output))

(l::Linear)(x) = l.w * x .+ l.b

# Sample from trained model

function generate(model,chars,n)
    function sample(y)
        p = Array(exp.(y)); r = rand()*sum(p)
        for j=1:length(p); (r -= p[j]) < 0 && return j; end
    end
    x = 1
    h = []
    for i=1:n
        y = model([x], hidden=h)
        x = sample(y)
        print(chars[x])
    end
    println()
end

@info("Loading Shakespeare data")
include(Knet.dir("data","gutenberg.jl"))
trn,tst,shake_chars1 = shakespeare()
shake_text = String(shake_chars1[vcat(trn,tst)])

@info("Loading Shakespeare model")
isfile("shakespeare.jld2") || download("http://people.csail.mit.edu/deniz/models/tutorial/shakespeare.jld2","shakespeare.jld2")
shake_model, shake_chars = Knet.load("shakespeare.jld2","model","chars")

@info("Reading Julia files")
base = joinpath(Sys.BINDIR, Base.DATAROOTDIR, "julia")
julia_text = ""
for (root,dirs,files) in walkdir(base)
    for f in files
        global julia_text
        f[end-2:end] == ".jl" || continue
        julia_text *= read(joinpath(root,f),String)
    end
    # println((root,length(files),all(f->contains(f,".jl"),files)))
end

@info("Loading Julia model")
isfile("juliacharlm.jld2") || download("http://people.csail.mit.edu/deniz/models/tutorial/juliacharlm.jld2","juliacharlm.jld2")
julia_model, julia_chars = Knet.load("juliacharlm.jld2","model","chars")


nothing
