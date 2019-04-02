# # Motivation
#
# Hello, and welcome! We're excited to be your gateway into machine learning. ML is a rapidly growing field that's buzzing with opportunity. Why? Because the tools and skills employed by ML specialists are extremely powerful and allow them to draw conclusions from large data sets quickly and with relative ease.
#
# Take the Celeste project, for example. This is a project that took 178 **tera**bytes of data on the visible sky and used it to catalogue 188 millions stars and galaxies. "Cataloguing" these stars meant identifying characteristics like their locations, colors, sizes, and morphologies. This is an amazing feat, *especially* because this entire calculation took under 15 minutes.
#
# <img src="data/Celeste.png" alt="Drawing" style="width: 1000px;"/>
#
# How are Celeste's calculations so fast? To achieve performance on this scale, the Celeste team uses the Julia programming language to write their software and supercomputers from Lawrence Berkeley National Lab's NERSC as their hardware. In this course, we unfortunately won't be able to give you access to a top 10 supercomputer, but we will teach you how to use Julia!
#
# We're confident that this course will put you on your way to understanding many of the important concepts and "buzz words" in ML. To get you started, we'll teach you how to teach a machine to tell the difference between images of apples and bananas, i.e to **classify** images as being one or the other type of fruit.
#
# Like Project Celeste, we'll use the [Julia programming language](https://julialang.org/) to do this. In particular, we'll be working in [Jupyter notebooks](http://jupyter.org/) like this one! (Perhaps you already know that the "ju" in Jupyter comes from Julia.)

#-

# ## What do the images we want to classify look like?
#
# We can use the `Images.jl` package in Julia to load sample images from this dataset. Most of the data we will use live in the `data` folder in this repository.

## using Pkg; Pkg.add(["Images", "ImageMagick"])
using Images  # To execute hit <shift> + enter

#-

apple = load("data/10_100.jpg")

#-

banana = load("data/104_100.jpg")

# The dataset consists of many images of different fruits, viewed from different positions.
# These images are [available on GitHub here](https://github.com/Horea94/Fruit-Images-Dataset).

#-

# ## What is the goal?

#-

# The ultimate goal is to feed one of these images to the computer and for it to identify whether the image represents an apple or a banana!  To do so, we will **train** the computer to learn **for itself** how to
# distinguish the two images.
#
# The following notebooks will walk you step by step through the underlying math and machine learning concepts you need to know in order to accomplish this classification.
#
# They alternate between two different types of notebooks: those labelled **ML** (Machine Learning), which are designed to give a high-level overview of the concepts we need for machine learning, but which gloss over some of the technical details; and those labelled **Tools**, which dive into the details of coding in Julia that will be key to actually implement the machine learning algorithms ourselves.
#
# The notebooks contain many **Exercises**. By doing these exercises in Julia, you will learn the basics of machine learning!

