server="ai.ku.edu.tr/"
if !isdir("data/demo")
    println("Downloading sample questions and images from CLEVR dataset...")
    download(server*"data/mac-network/demo.tar.gz","demo.tar.gz")
    run(`tar -xzf demo.tar.gz`)
    rm("demo.tar.gz")
end

if !isfile("models/macnet.jld2")
    println("Downloading pre-trained model from our servers...")
    download(server*"models/mac-network/macnet.jld2","models/macnet.jld2")
end
println("Demo setup is completed")
