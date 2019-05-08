# # Image classification with very deep convolutional networks

using Knet
cd("vgg")
include("vgg.jl")

#-

@doc VGG

#-

VGG.main("--help")

#-

using FileIO
img = download("http://home.mweb.co.za/pa/pak04857/uniweb/animalimages/elephantthumb.jpg")
load(img)

#-

VGG.main(img)

#-



