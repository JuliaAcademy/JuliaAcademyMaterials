# # Julia is fast
#
# Very often, benchmarks are used to compare languages.  These benchmarks can
# lead to long discussions, first as to exactly what is being benchmarked and
# secondly what explains the differences.  These simple questions can sometimes
# get more complicated than you at first might imagine.
#
# The purpose of this notebook is for you to see a simple benchmark for
# yourself.
#
# (This material began life as a wonderful lecture by Steven Johnson at MIT:
# [Boxes and registers](https://github.com/stevengj/18S096/blob/master/lectures/lecture1/Boxes-and-registers.ipynb)).

#-

# # Outline of this notebook
#
# - Define the sum function
# - Implementations & benchmarking of sum in...
#     - Julia (built-in)
#     - Julia (hand-written)
#     - C (hand-written)
#     - python (built-in)
#     - python (numpy)
#     - python (hand-written)
# - Towards exploiting parallelism with Julia
#     - Allowing for floating point associativity
#     - Making use of four cores at once: built-in
#     - Making use of four cores at once: hand-written
# - Summary of benchmarks

#-

# # `sum`: An easy enough function to understand

#-

# Consider the  **sum** function `sum(a)`, which computes
# $$
# \mathrm{sum}(a) = \sum_{i=1}^n a_i,
# $$
# where $n$ is the length of `a`.

a = rand(10^7) # 1D vector of random numbers, uniform on [0,1)

#-

sum(a)

# The expected result is ~0.5 * 10^7, since the mean of each entry is 0.5

#-

# # Benchmarking a few ways in a few languages

@time sum(a)

#-

@time sum(a)

#-

@time sum(a)

# The `@time` macro can yield noisy results, so it's not our best choice for benchmarking!
#
# Luckily, Julia has a `BenchmarkTools.jl` package to make benchmarking easy and accurate:

## using Pkg
## Pkg.add("BenchmarkTools")

#-

using BenchmarkTools

#-

@benchmark sum($a)

# # 1. Julia Built-in
#
# So that's the performance of Julia's built-in sum — but that could be doing any number of tricks to be fast, including not using Julia at all in the first place! Of course, it is indeed written in Julia, but would it perform if we write a naive implementation ourselves?

@which sum(a)

# Let's save these benchmark results to a dictionary so we can start keeping track of them and comparing them down the line.

j_bench = @benchmark sum($a)

#-

d = Dict()
d["Julia built-in"] = minimum(j_bench.times) / 1e6
d

# # 2. Julia (hand-written)

function mysum(A)
    s = 0.0
    for a in A
        s += a
    end
    return s
end

#-

j_bench_hand = @benchmark mysum($a)

#-

d["Julia hand-written"] = minimum(j_bench_hand.times) / 1e6
d

# So that's about 2x slower than the builtin definition. We'll see why later on.
#
# But first: is this fast?  How would we know?  Let's compare it to some other languages...

#-

# #  3. The C language
#
# C is often considered the gold standard: difficult on the human, nice for the machine. Getting within a factor of 2 of C is often satisfying. Nonetheless, even within C, there are many kinds of optimizations possible that a naive C writer may or may not get the advantage of.
#
# The current author does not speak C, so he does not read the cell below, but is happy to know that you can put C code in a Julia session, compile it, and run it. Note that the `"""` wrap a multi-line string.

using Libdl
C_code = """
    #include <stddef.h>
    double c_sum(size_t n, double *X) {
        double s = 0.0;
        for (size_t i = 0; i < n; ++i) {
            s += X[i];
        }
        return s;
    }
"""

const Clib = tempname()   # make a temporary file


## compile to a shared library by piping C_code to gcc
## (works only if you have gcc installed):

open(`gcc -fPIC -O3 -msse3 -xc -shared -o $(Clib * "." * Libdl.dlext) -`, "w") do f
    print(f, C_code)
end

## define a Julia function that calls the C function:
c_sum(X::Array{Float64}) = ccall(("c_sum", Clib), Float64, (Csize_t, Ptr{Float64}), length(X), X)

#-

c_sum(a)

#-

c_sum(a) ≈ sum(a) # type \approx and then <TAB> to get the ≈ symbolb

# We can now benchmark the C code directly from Julia:

c_bench = @benchmark c_sum($a)

#-

d["C"] = minimum(c_bench.times) / 1e6  # in milliseconds
d

# # 4. Python's built in `sum`

#-

# The `PyCall` package provides a Julia interface to Python:

## using Pkg; Pkg.add("PyCall")
using PyCall

#-

## get the Python built-in "sum" function:
pysum = pybuiltin("sum")

#-

pysum(a)

#-

pysum(a) ≈ sum(a)

#-

py_list_bench = @benchmark $pysum($a)

#-

d["Python built-in"] = minimum(py_list_bench.times) / 1e6
d

# # 5. Python: `numpy`
#
# `numpy` is an optimized C library, callable from Python.
# It may be installed within Julia as follows:

## using Pkg; Pkg.add("Conda")
using Conda

#-

## Conda.add("numpy")

#-

numpy_sum = pyimport("numpy")["sum"]

py_numpy_bench = @benchmark $numpy_sum($a)

#-

numpy_sum(a)

#-

numpy_sum(a) ≈ sum(a)

#-

d["Python numpy"] = minimum(py_numpy_bench.times) / 1e6
d

# # 6. Python, hand-written

py"""
def py_sum(A):
    s = 0.0
    for a in A:
        s += a
    return s
"""

sum_py = py"py_sum"

#-

py_hand = @benchmark $sum_py($a)

#-

sum_py(a)

#-

sum_py(a) ≈ sum(a)

#-

d["Python hand-written"] = minimum(py_hand.times) / 1e6
d

# # Summary so far

for (key, value) in sort(collect(d), by=last)
    println(rpad(key, 25, "."), lpad(round(value; digits=1), 6, "."))
end

# We seem to have three different performance classes here: The numpy and Julia
# builtins are leading the pack, followed by the hand-written Julia and C
# definitions. Those seem to be about 2x slower.  And then we have the much much
# slower Python definitions over 100x slower.

#-

# # Exploiting parallelism with Julia

#-

# The fact that our hand-written Julia solution was almost an even multiple of
# 2x slower than the builtin solutions is a big clue: perhaps theres some sort
# of 2x parallelism going on here?
#
# (In fairness, there are ways to exploit parallelism in other languages, too,
# but for brevity we won't cover them)

#-

# # 7. Julia (allowing floating point associativity)

#-

# The `for` loop
#
# ```julia
# for a in A
#     s += a
# end
# ```
#
# defines a very strict _order_ to the summation: Julia follows exactly what you
# wrote and adds the elements of `A` to the result `s` in the order it iterates.
# Since floating point numbers aren't associative, a rearrangement here would
# change the answer — and Julia is loathe to give you different answer than
# the one you asked for.
#
# You can, however, tell Julia to relax that rule and allow for associativity
# with the `@fastmath` macro. This might allow Julia to rearrange the sum in an
# advantageous manner.

function mysum_fast(A)
    s = 0.0
    for a in A
        @fastmath s += a
    end
    s
end

#-

j_bench_hand_fast = @benchmark mysum_fast($a)

#-

mysum_fast(a)

#-

d["Julia hand-written fast"] = minimum(j_bench_hand_fast.times) / 1e6
d

# # 8. Distributed Julia (built-in)

#-

# We can take this one step further: nearly every modern computer these days has
# multiple cores. All the above solutions are working one core hard, but all the
# others are just sitting by idly. Let's put them to work!

using Distributed
using DistributedArrays
addprocs(4)
@sync @everywhere workers() include("/opt/julia-1.0/etc/julia/startup.jl") # Needed just for JuliaBox
@everywhere using DistributedArrays

#-

adist = distribute(a)
j_bench_dist = @benchmark sum($adist)

#-

d["Julia 4x built-in"] = minimum(j_bench_dist.times) / 1e6
d

# # 9. Distributed Julia (hand-written)
#
# Ok, that might be cheating, too — it's again just calling a library
# function. Is it possible to write distributed sum ourselves?

function mysum_dist(a::DArray)
    r = Array{Future}(undef, length(procs(a)))
    for (i, id) in enumerate(procs(a))
        r[i] = @spawnat id sum(localpart(a))
    end
    return sum(fetch.(r))
end

#-

j_bench_hand_dist = @benchmark mysum_dist($adist)

#-

d["Julia 4x hand-written"] = minimum(j_bench_hand_dist.times) / 1e6
d

# # Overall Summary

for (key, value) in sort(collect(d), by=last)
    println(rpad(key, 25, "."), lpad(round(value; digits=1), 6, "."))
end

# # Key take-aways
#
# * Julia allows for serial C-like performance, even with hand-written functions
# * Julia allows us to exploit many forms of parallelism to further improve performance. We demonstrated:
#     * Single-processor parallelism with SIMD
#     * Multi-process parallelism with DistributedArrays
# * But there are many other ways to express parallelism, too!

