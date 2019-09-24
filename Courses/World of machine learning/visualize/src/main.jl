#This implementation is very similar to original implementation in https://github.com/stanfordnlp/mac-network
using JSON,Knet,Images,HDF5,Plots
#import AutoGrad: cat1d
using Printf,Random
include("model.jl")
savemodel(filename,m,mrun,o) = Knet.save(filename,"m",m,"mrun",mrun,"o",o)

function loadmodel(filename;onlywrun=false)
    d = Knet.load(filename)
    if onlywrun
        mrun=d["mrun"];o=d["o"]
        m=nothing
    else
        m=d["m"];o=d["o"];mrun=d["mrun"];
    end
    return m,mrun,o;
end

function getQdata(dhome,set)
    JSON.parsefile(dhome*set*".json")
end

function invert(vocab)
       int2tok = Array{String}(undef,length(vocab))
       for (k,v) in vocab; int2tok[v] = k; end
       return int2tok
end

function getDicts(dhome,dicfile)
    dic  = JSON.parsefile(dhome*dicfile*".json")
    qvoc = dic["word_dic"]
    avoc = dic["answer_dic"]
    i2w  = invert(qvoc)
    i2a  = invert(avoc)
    return qvoc,avoc,i2w,i2a
end

function loadFeatures(dhome,set;h5=false)
    if h5
        return h5open(dhome*set*".hdf5","r")["data"]
    else
        feats = reinterpret(Float32,read(open(dhome*set*".bin")))
        return reshape(feats,(14,14,1024,div(length(feats),200704)))
    end
end

function miniBatch(data;shfl=true,srtd=false,B=32)
    L = length(data)
    shfl && shuffle!(data)
    srtd && sort!(data;by=x->length(x[2]))
    batchs = [];
    for i=1:B:L
        b         = min(L-i+1,B)
        questions = Any[]
        answers   = zeros(Int,b)
        images    = Any[]
        families  = zeros(Int,b)

        for j=1:b
            crw = data[i+j-1]
            push!(questions,reverse(Array{Int}(crw[2])))
            push!(images,parse(Int,crw[1][end-9:end-4])+1)
            answers[j]  = crw[3]
            families[j] = crw[4]
        end

        lngths     = length.(questions);
        srtindices = sortperm(lngths;rev=true)

        lngths     = lngths[srtindices]
        Tmax       = lngths[1]
        questions  = questions[srtindices]
        answers    = answers[srtindices]
        images     = images[srtindices]
        families   = families[srtindices]

        qs = Int[];
        batchSizes = Int[];
        pads = falses(b,Tmax)

        for k=1:b
           pads[k,lngths[k]+1:Tmax].=true
        end

        if sum(pads)==0
           pads=nothing
        end

        while true
            batch = 0
            for j=1:b
                if length(questions[j]) > 0
                    batch += 1
                    push!(qs,pop!(questions[j]))
                end
            end
            if batch != 0
                push!(batchSizes,batch)
            else
                break;
            end
        end
        push!(batchs,(images,qs,answers,batchSizes,pads,families))
    end
    return batchs
end

function loadTrainingData(dhome="data/";h5=false)
    !h5 && println("Loading pretrained features for train&val sets.
                It requires minimum 70GB RAM!!!")
    trnfeats = loadFeatures(dhome,"train";h5=h5)
    valfeats = loadFeatures(dhome,"val";h5=h5)
    println("Loading questions ...")
    trnqstns = getQdata(dhome,"train")
    valqstns = getQdata(dhome,"val")
    println("Loading dictionaries ... ")
    qvoc,avoc,i2w,i2a = getDicts(dhome,"dic")
    return (trnfeats,valfeats),(trnqstns,valqstns),(qvoc,avoc,i2w,i2a)
end

function loadDemoData(dhome="data/demo/")
    println("Loading demo features ...")
    feats = loadFeatures(dhome,"demo")
    println("Loading demo questions ...")
    qstns = getQdata(dhome,"demo")
    println("Loading dictionaries ...")
    dics = getDicts(dhome,"dic")
    return feats,qstns,dics
end

function modelrun(M,data,feats,o,Mrun=nothing;train=false)
    getter(id) = view(feats,:,:,:,id)
    cnt=total=0.0; L=length(data);
    Mparams   = params(M)
    Rparams   = Mrun !== nothing ? params(Mrun) : nothing
    # results   = similar(Array{Float32},200704*48) #uncomment for INPLACE
    println("Timer Starts");
    for i in 1:L
        ids,questions,answers,batchSizes,pad,families = data[i]
        B    = batchSizes[1]
        xB   = atype(ones(Float32,1,B))
        #x    = inplace_batcher(results,feats,ids) #uncomment for INPLACE
        x    = batcher1(feats,ids) #comment for INPLACE
        xS   = atype(x)
        #xS   = atype(reshape(cat1d(map(getter,ids)...),14,14,1024,B))
        xP   = pad==nothing ? nothing : atype(pad*Float32(1e22))
        if train
            J = @diff M(questions,batchSizes,xS,xB,xP;answers=answers,p=o[:p],selfattn=o[:selfattn],gating=o[:gating])
            cnt += value(J)*B; total += B;
            for w in Mparams
                update!(w.value,grad(J,w),w.opt)
            end
            if Mrun != nothing
                for (wr,wi) in zip(Rparams,Mparams);
                    axpy!(1.0f0-o[:ema],wi.value-wr.value,wr.value);
                end
            end
        else
            preds  = M(questions,batchSizes,xS,xB,xP;p=o[:p],selfattn=o[:selfattn],gating=o[:gating])
            cnt   += sum(preds.==answers)
            total += B
        end
        i % 1000 == 0 && println(@sprintf("%.2f Accuracy|Loss", train ? cnt/total : 100cnt/total))
    end
    train && savemodel(o[:prefix]*".jld2",M,Mrun,o);
end

function train!(M,Mrun,sets,feats,o)
    @info "Training Starts...."
    setoptim!(M,o)
    for i=1:o[:epochs]
        println("Epoch $(i) starts...")
        modelrun(M,sets[1],feats[1],o,Mrun;train=true)
        modelrun(Mrun,sets[2],feats[2],o;train=false)
    end
    return M,Mrun;
end

function train(sets,feats,o)
     if o[:mfile]==nothing
         M    = MACNetwork(o);
         Mrun = deepcopy(M)
     else
         M,Mrun,o = loadmodel(o[:mfile])
     end
     train!(M,Mrun,sets,feats,o)
     return M,Mrun;
end

function train(dhome="data/",o=nothing)
     if o==nothing
         o=Dict(:h5=>false,:mfile=>nothing,:epochs=>10,
                :lr=>0.0001,:p=>12,:ema=>0.999f0,:batchsize=>32,
                :selfattn=>false,:gating=>false,:d=>512,
                :shuffle=>true,:sorted=>false,:prefix=>string(now())[1:10],
                :vocab_size=>90,:embed_size=>300, :dhome=>"data/", :loadresnet=>false)
     end
     feats,qdata,dics = loadTrainingData(dhome;h5=o[:h5])
     sets = []
     for q in qdata; push!(sets,miniBatch(q;shfle=o[:shuffle],srtd=o[:sorted])); end
     qdata = nothing; #gc();
     M,Mrun = train(sets,feats,o)
     return M,Mrun,sets,feats,dics;
end

function validate(Mrun,valset,valfeats,o)
     modelrun(Mrun,valset,valfeats,o;train=false)
end

function validate(mfile,valset,valfeats,o)
     _,Mrun,_ = loadmodel(mfile)
     modelrun(Mrun,valset,valfeats;train=false)
     return Mrun
end

function validate(mfile,dhome,o)
     _,Mrun,_,o   = loadmodel(mfile)
     valfeats     = loadFeatures(dhome,"val")
     qdata        = getQdata(dhome,"val")
     dics         = getDicts(dhome,"dic")
     valset       = miniBatch(qdata;shfle=o[:shuffle],srtd=o[:sorted])
     modelrun(Mrun,valset,valfeats,o;train=false)
     return Mrun,valset,valfeats
end

function singlerun(Mrun,feat,question;p=12,selfattn=false,gating=false)
    results        = Dict{String,Any}("cnt"=>1)
    batchSizes     = ones(Int,length(question))
    xB             = atype(ones(Float32,1,1))
    outputs = Mrun(question,batchSizes,feat,xB,nothing;tap=results,p=p,selfattn=selfattn,gating=gating,allsteps=true)
    prediction = argmax(results["y"])
    return results,prediction,outputs
end

function visualize(img,results;p=12)
    s_y,s_x = size(img)./14
    for k=1:p
        α = results["w_attn_$(k)"][:]
        wrds    = i2w[question]
        p = bar(α;xticks=(collect(1:length(wrds)),wrds),xrotation=90,bar_width = 1,
            xtickfont = font(8, "Courier"),yticks=0:.1:(maximum(α)+.1),
            legend=false,size=(600,100+400*(maximum(α))),aspect_ratio=10)      
        savefig(p,"plots/$(k).png")
        display(RGB.(load("plots/$(k).png")))
        println("Image Attention Map: ")
        flush(stdout)
        hsvimg = HSV.(img);
        attn = results["KB_attn_$(k)"]
        for i=1:14,j=1:14
            rngy          = floor(Int,(i-1)*s_y+1):floor(Int,min(i*s_y,320))
            rngx          = floor(Int,(j-1)*s_x+1):floor(Int,min(j*s_x,480))
            hsvimg[rngy,rngx]  = scalepixel.(hsvimg[rngy,rngx],attn[LinearIndices((1:14,1:14))[i,j]])
        end
        display(hsvimg)
    end
end

function scalepixel(pixel,scaler)
     return HSV(pixel.h,pixel.s,min(1.0,pixel.v+5*scaler))
end
