using Knet, Random
if !isdefined(Main,:atype)
    global atype = gpu() < 0 ? Array{Float32} : KnetArray{Float32}
end
abstract type Model;end;
struct ResNet <: Model; w; end
function (M::ResNet)(m,imgurl::String,avgimg;stage=3)
    img = imgdata(imgurl, avgimg)
    return M(w,m,atype(img);stage=stage);
end
function ResNet(atype::Type;stage=3)
    w,m,meta = ResNetLib.resnet101init(;trained=true,stage=stage)
    global avgimg = meta["normalization"]["averageImage"]
    global descriptions = meta["classes"]["description"]
    return w,m,meta,avgimg
end
ResNet() = ResNet(nothing);

Par(x)     = Param(atype(x))
init(o...) = Par(xavier(Float32,o...))
bias(o...) = Par(zeros(Float32,o...))
#elu(x)     = relu.(x) + (exp.(min.(0,x)) .- 1.0f0)
#softmax(x,dims) = exp.(logp(x;dims=dims))

struct Linear <: Model; w; b; end
(m::Linear)(x) = m.w * x .+ m.b
Linear(input::Int,output::Int;winit=init,binit=bias) = Linear(winit(output,input),binit(output,1))
Linear() = Linear(nothing,nothing)

struct Embed <: Model; w ; end
(m::Embed)(x)  = m.w[:,x]
Embed(input,embed;winit=rand) = Embed(Par(winit(Float32,embed,input)))
Embed() = Embed(nothing)

struct Conv4D <: Model; w; b; end
(m::Conv4D)(x) = conv4(m.w,x;padding=1,stride=1) .+ m.b
Conv4D(h,w,c,o;winit=init,binit=bias) = Conv4D(winit(h,w,c,o),binit(1,1,o,1))

struct CNN <: Model
    layer1::Conv4D
    layer2::Conv4D
end

function (m::CNN)(x;train=false)
    if train;x=dropout(x,0.18);end
    x1 = elu.(m.layer1(x))
    if train;x1=dropout(x1,0.18);end
    x2 = elu.(m.layer2(x1))
    h,w,c,b = size(x2)
    permutedims(reshape(x2,h*w,c,b),(2,3,1))
end

CNN(h,w,c,d;winit=init,binit=bias) = CNN(Conv4D(h,w,c,d),Conv4D(h,w,d,d))

struct mRNN  <: Model
    rnn
end

function (m::mRNN)(x;batchSizes=[1],train=false)
    B = first(batchSizes)
    if last(batchSizes)!=B
        y,hyout,_,_ = rnnforw(m.rnn,m.rnn.w,x;batchSizes=batchSizes,hy=true,cy=false)
    else
        x           = reshape(x,size(x,1),B,div(size(x,2),B))
        y,hyout,_,_ = rnnforw(m.rnn,m.rnn.w,x;hy=true,cy=false)
    end
    return y,hyout
end
mRNN(input::Int,hidden::Int;o...) = mRNN(RNN(input, hidden;o...))

struct QUnit  <: Model
    embed::Embed
    rnn::mRNN
    linear::Linear
end
function (m::QUnit)(x;batchSizes=[1],train=false)
    xe = m.embed(x)
    if train; xe=dropout(xe,0.15); end;
    y,hyout = m.rnn(xe;batchSizes=batchSizes)
    q            = vcat(hyout[:,:,1],hyout[:,:,2])
    if train; q=dropout(q,0.08); end;
    B = batchSizes[1]
    if ndims(y) == 2
        indices      = bs2ind(batchSizes)
        lngths       = length.(indices)
        Tmax         = maximum(lngths)
        td,B         = size(q)
        d            = div(td,2)
        cw           = Any[];
        for i=1:length(indices)
            y1 = y[:,indices[i]]
            df = Tmax-lngths[i]
            if df > 0
                cpad = zeros(Float32,2d*df) # zeros(Float32,2d,df)
                kpad = atype(cpad)
                ypad = reshape(cat1d(y1,kpad),2d,Tmax) # hcat(y1,kpad)
                push!(cw,ypad)
            else
                push!(cw,y1)
            end
        end
        cws_2d =  reshape(vcat(cw...),2d,B*Tmax)
    else
        d      = div(size(y,1),2)
        Tmax   = size(y,3)
        cws_2d = reshape(y,2d,B*Tmax)
    end
    cws_3d =  reshape(m.linear(cws_2d),(d,B,Tmax))
    return q,cws_3d;
end
QUnit(vocab::Int,embed::Int,hidden::Int;bidir=true) = QUnit(Embed(vocab,embed),
                                                            mRNN(embed,hidden;bidirectional=bidir),
                                                            Linear(2hidden,hidden))

function bs2ind(batchSizes)
    B = batchSizes[1]
    indices = Any[]
    for i=1:B
        ind = i.+cumsum(filter(x->(x>=i),batchSizes)[1:end-1])
        push!(indices,append!(Int[i],ind))
    end
    return indices
end

struct Control  <: Model
    cq::Linear
    att::Linear
end
function (m::Control)(c,q,cws,pad;train=false,tap=nothing)
      d,B,T = size(cws)
      cqi   = reshape(m.cq(vcat(c,q)),(d,B,1))
      cvis  = reshape(cqi .* cws,(d,B*T))
      cvis_2d = reshape(m.att(cvis),(B,T)) #eq c2.1.2
      if pad != nothing
          cvi = reshape(softmax(cvis_2d .- pad,dims=2),(1,B,T)) #eq c2.2
      else
          cvi = reshape(softmax(cvis_2d,dims=2),(1,B,T)) #eq c2.2
      end
      tap!=nothing && get!(tap,"w_attn_$(tap["cnt"])",Array(reshape(cvi,B,T)))
      cnew = reshape(sum(cvi.*cws;dims=3),(d,B))
end
Control(d::Int) = Control(Linear(2d,d),Linear(d,1))

struct Read  <: Model
    me::Linear
    Kbe::Linear
    Kbe2::Linear
    Ime
    att::Linear
end

function (m::Read)(mp,ci,cws,KBhw′,KBhw′′;train=false,tap=nothing)
    d,B,N = size(KBhw′); BN = B*N
    mi_3d = reshape(m.me(mp),(d,B,1))
    ImKB  = reshape(mi_3d .* KBhw′,(d,BN)) # eq r1.2
    ImKB′ = reshape(elu.(m.Ime*ImKB .+ KBhw′′),(d,B,N)) #eq r2
    ci_3d = reshape(ci,(d,B,1))
    IcmKB_pre = elu.(reshape(ci_3d .* ImKB′,(d,BN))) #eq r3.1.1
    if train; IcmKB_pre = dropout(IcmKB_pre,0.15); end;
    IcmKB = reshape(m.att(IcmKB_pre),(B,N)) #eq r3.1.2
    mvi = reshape(softmax(IcmKB,dims=2),(1,B,N)) #eq r3.2
    tap!=nothing && get!(tap,"KB_attn_$(tap["cnt"])",Array(reshape(mvi,B,N)))
    mnew = reshape(sum(mvi.*KBhw′;dims=3),(d,B)) #eq r3.3
end
Read(d::Int) = Read(Linear(d,d),Linear(d,d),Linear(d,d),init(d,d),Linear(d,1))

struct Write  <: Model
    me::Linear
    cproj::Linear
    att::Linear
    mpp
    gating::Linear
end

function (m::Write)(m_new,mi₋1,mj,ci,cj;train=false,selfattn=true,gating=true,tap=nothing)
    d,B        = size(m_new)
    T          = length(mj)
    mi         = m.me(vcat(m_new,mi₋1))
    !selfattn && return mi
    ciproj     = m.cproj(ci)
    ci_3d      = reshape(ciproj,d,B,1)
    cj_3d      = reshape(cat1d(cj...),(d,B,T)) #reshape(hcat(cj...),(d,B,T)) #
    sap        = reshape(ci_3d.*cj_3d,(d,B*T)) #eq w2.1.1
    sa         = reshape(m.att(sap),(B,T)) #eq w2.1.2
    sa′        = reshape(softmax(sa,dims=2),(1,B,T)) #eq w2.1.3
    mj_3d      = reshape(cat1d(mj...),(d,B,T)) #reshape(hcat(mj...),(d,B,T)) #
    mi_sa      = reshape(sum(sa′ .* mj_3d;dims=3),(d,B))
    mi′′       = m.mpp*mi_sa .+ mi #eq w2.3
    !gating && return mi′′
    σci′       = sigm.(m.gating(ci))  #eq w3.1
    mi′′′      = (σci′ .* mi₋1) .+  ((1 .- σci′) .* mi′′) #eq w3.2
end

function Write(d::Int;selfattn=true,gating=true)
    if selfattn
        if gating
            Write(Linear(2d,d),Linear(d,d),Linear(d,1),init(d,d),Linear(d,1))
        else
            Write(Linear(2d,d),Linear(d,d),Linear(d,1),init(d,d),Linear())
        end
    else
        Write(Linear(2d,d),Linear(),Linear(),nothing,Linear())
    end
end

struct MAC <: Model
    control::Control
    read::Read
    write::Write
end
function (m::MAC)(qi,cws,mi,mj,ci,cj,KBhw′,KBhw′′,pad;train=false,selfattn=true,gating=true,tap=nothing)
    cnew = m.control(ci,qi,cws,pad;train=train,tap=tap)
    ri   = m.read(mi,ci,cws,KBhw′,KBhw′′;train=train,tap=tap)
    mnew = m.write(ri,mi,mj,ci,cj;train=train,selfattn=selfattn,gating=gating)
    return cnew,mnew
end
MAC(d::Int;selfattn=false,gating=false) = MAC(Control(d),Read(d),Write(d))

struct Output <: Model
    qe::Linear
    l1::Linear
    l2::Linear
end

function (m::Output)(q,mp;train=false)
  qe = elu.(m.qe(q))
  x  = elu.(m.l1(cat(qe,mp;dims=1)))
  m.l2(x)
end
Output(d::Int) = Output(Linear(2d,d),Linear(2d,d),Linear(d,28))

struct MACNetwork <: Model
    resnet::ResNet
    cnn::CNN
    qunit::QUnit
    qindex::Linear
    mac::MAC
    output::Output
    c0
    m0
end

function (M::MACNetwork)(qs,batchSizes,xS,xB,xP;answers=nothing,p=12,selfattn=false,gating=false,tap=nothing,allsteps=false)
    train         = answers!=nothing
    #STEM Processing
    KBhw          = M.cnn(xS;train=train)
    #Read Unit Precalculations
    d,B,N         = size(KBhw)
    KBhw_2d       = reshape(KBhw,(d,B*N))
    if train; KBhw_2d = dropout(KBhw_2d,0.15); end;
    KBhw′_pre     = M.mac.read.Kbe(KBhw_2d) # look if it is necessary
    KBhw′′        = M.mac.read.Kbe2(KBhw′_pre)
    KBhw′         = reshape(KBhw′_pre,(d,B,N))

    #Question Unit
    q,cws         = M.qunit(qs;batchSizes=batchSizes,train=train)
    qi_c          = M.qindex(q)
    #Memory Initialization
    ci            = M.c0*xB
    mi            = M.m0*xB

    if selfattn
        cj=[ci]; mj=[mi]
    else
        cj=nothing; mj=nothing
    end

    for i=1:p
        qi        = qi_c[(i-1)*d+1:i*d,:]
        if train; ci = dropout(ci,0.15); mi = dropout(mi,0.15); end
        ci,mi = M.mac(qi,cws,mi,mj,ci,cj,KBhw′,KBhw′′,xP;train=train,selfattn=selfattn,gating=gating,tap=tap)
        if selfattn; push!(cj,ci); push!(mj,mi); end
        tap!=nothing && (tap["cnt"]+=1)
    end

    y = M.output(q,mi;train=train)    

    if answers==nothing
        predmat = convert(Array{Float32},y)
        tap!=nothing && get!(tap,"y",predmat)
        predictions = mapslices(argmax,predmat,dims=1)[1,:]
        if allsteps
            outputs = []
            for i=1:p-1
                yi = M.output(q,mj[i];train=train)
                yi = convert(Array{Float32},yi)
                push!(outputs,mapslices(argmax,yi,dims=1)[1,:])
            end
            push!(outputs,predictions)
            return outputs
        end
        return predictions
    else
        return nll(y,answers)
    end
end

function MACNetwork(o::Dict)
           MACNetwork(ResNet(),
                      CNN(3,3,1024,o[:d]),
                      QUnit(o[:vocab_size],o[:embed_size],o[:d]),
                      Linear(2*o[:d],o[:p]*o[:d]),
                      MAC(o[:d];selfattn=o[:selfattn],gating=o[:gating]),
                      Output(o[:d]),
                      init(o[:d],1), Par(randn(Float32,o[:d],1)))
end

function setoptim!(m::MACNetwork,o)
    for param in Knet.params(m)
        param.opt = Adam(;lr=o[:lr])
    end
end

function benchmark(M::MACNetwork,feats,o;N=10)
    getter(id) = view(feats,:,:,:,id)
    B=32;L=25
    @time for i=1:N
        ids  = randperm(128)[1:B]
        xB   = atype(ones(Float32,1,B))
        xS   = atype(batcher(map(getter,ids)))
        xQ   = [rand(1:84) for i=1:B*L]
        answers = [rand(1:28) for i=1:B]
        batchSizes = [B for i=1:L]
        xP   = nothing
        y    = @diff M(xQ,batchSizes,xS,xB,xP;answers=answers,p=o[:p],selfattn=o[:selfattn],gating=o[:gating])
    end
end
function benchmark(feats,o;N=30)
    M     = MACNetwork(o);
    benchmark(M,feats,o;N=N)
end


const feats_L = 200704;
const feats_H = 14;
const feats_C = 1024;
function batcher1(feats,args)
    B = length(args)
    totlen = feats_L*B
    result = similar(Array{Float32}, totlen)
    starts = (0:B-1) .*feats_L .+ 1; ends = starts .+ feats_L .- 1;
    for i=1:B
        result[starts[i]:ends[i]] = view(feats,:,:,:,args[i])
    end
    return reshape(result,feats_H,feats_H,feats_C,B)
end

function inplace_batcher(result,data,args)
     B = length(args)
     totlen = feats_L*B
     starts = (0:B-1) .* feats_L .+ 1; ends = starts .+ feats_L .- 1;
     for i=1:B
          result[starts[i]:ends[i]] = view(data,:,:,:,args[i])
     end
     return reshape(result,feats_H,feats_H,feats_C,B)
end
