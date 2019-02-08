# # Parallel Algorithms: Thinking in Parallel
#
# Now that we're starting to see the challenges of parallelism, it's worth taking
# a step back and examining how we might go about designing parallel algorithms.
#
# This is adapted from a [workshop paper](http://jiahao.github.io/parallel-prefix/) by Jiahao Chen and
# Alan Edelman entitled "Parallel Prefix Polymorphism Permits Parallelization, Presentation & Proof" and
# will appear in the proceedings of the [First Workshop for High Performance Technical Computing in Dynamic
# Languages](http://jiahao.github.io/hptcdl-sc14/), held in conjunction with [SC14: The International Conference on High Performance Computing, Networking, Storage and Analysis](http://sc14.supercomputing.org/)
#

using Compose, Gadfly, Interact

# # `reduce()`
#
# Reduction applies a binary operator to a vector repeatedly to return a scalar. Thus + becomes sum, and * becomes prod.
#
# It is considered a basic parallel computing primitive.
#

reduce(+, 1:8), sum(1:8)  # triangular numbers

#-

reduce(*, 1:8), prod(1:8) # factorials

#-

boring(a,b)=a
@show reduce(boring, 1:8)
boring2(a,b)=b
@show reduce(boring2, 1:8)

# You can also use reduce to compute Fibonacci numbers using their recurrences.
#
# $$\begin{pmatrix} f_2 \\f_1 \end{pmatrix} = \begin{pmatrix} f_1 + f_0 \\ f_1 \end{pmatrix}
# = \begin{pmatrix} 1 & 1 \\ 1 & 0 \end{pmatrix} \begin{pmatrix} f_1 \\ f_0 \end{pmatrix} $$
#
# $$\begin{pmatrix} f_{n+1} \\ f_n \end{pmatrix} = \dots
# = \begin{pmatrix} 1 & 1 \\ 1 & 0 \end{pmatrix}^n \begin{pmatrix} f_1 \\ f_0 \end{pmatrix} $$
#
# From this, you can show that
#
# $$\begin{pmatrix} 1 & 1 \\ 1 & 0 \end{pmatrix}^n  = \begin{pmatrix} f_{n+1} & f_n \\ f_n & f_{n-1} \end{pmatrix} $$
#
# (this applies reduce to the same argument over and over again -- there are of course other ways)

M=[1 1; 1 0]
reduce(*,fill(M,3))
prod(fill(M,3))

#-

@manipulate for n=1:100
    prod(fill(big.(M),n))
end

#-

fib(j)=reduce(*, fill(M,j))
fib.([4,7])


# You can solve recurrences of any complexity using `reduce`. For example, `reduce` can compute a Hadamard matrix from its definition in terms of its submatrices:
#
# $$H_2 = \begin{pmatrix} H_1 & H_1 \\ H_1 & -H_1 \end{pmatrix} = \begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix} \otimes H_1$$
#
# and so on.
#
# (Note: this is just using reduce to compute a matrix power.
# One can think of alternative ways for sure.)

## If A is m x n
## If B is p x q
## then kron(A,B) is mp x nq and has all the elements of A times all of the elements of B

#-

A=[1 2;3 4]
B=[10 100; 1 -10]
⊗(A,B)=kron(A,B)

M=[ 1 1;1 -1]
H=⊗(⊗(⊗(M,M),M),M)

#-

H'H

#-

Hadamard(n)=reduce(⊗, fill(M,n))
H=Hadamard(4)

#-

using LinearAlgebra
cumsum(1:8)  # It is useful to know that cumsum is a linear operator
## You can use power method! Below is the underlying matrix
A=tril(ones(Int,8,8))

#-

[A*(1:8),cumsum(1:8)]

# # `prefix`

#-

# Having discussed `reduce`, we are now ready for the idea behind prefix sum.
# Prefix or scan is long considered an important parallel
# primitive as well.
#
# Suppose you wanted to compute the partial sums of a vector, i.e. given
# `y[1:n]`, we want to overwrite the vector `y` with the vector of partial sums
#
# ```julia
# new_y[1] = y[1]
# new_y[2] = y[1] + y[2]
# new_y[3] = y[1] + y[2] + y[3]
# ...
# ```
#
# At first blush, it seems impossible to parallelize this, since
#
# ```julia
# new_y[1] = y[1]
# new_y[2] = new_y[1] + y[2]
# new_y[3] = new_y[2] + y[3]
# ...
# ```
#
# which appears to be an intrinsically serial process. As written with a `+`
# operator, this is `cumsum` — but note that it can generalize to any operation.

function prefix_serial!(y, ⊕)
    for i=2:length(y)
        y[i] = y[i-1] ⊕ y[i]
    end
    y
end

#-

prefix_serial!([1:8;],+)

#-

cumsum(1:8)

#-

prefix_serial!([1:8;], *)

#-

cumprod(1:8)

# However, it turns out that because these operations are associative, we can regroup the _order_ of how these sums or products are carried out. (This of course extends to other associative operations, too.) Another ordering of 8 associative operations is provided by `prefix8!`:

## Magic :)
function prefix8!(y, ⊕)
    length(y)==8 || error("length 8 only")
    for i in (2,4,6,8); y[i] = y[i-1] ⊕ y[i]; end
    for i in (  4,  8); y[i] = y[i-2] ⊕ y[i]; end
    for i in (      8); y[i] = y[i-4] ⊕ y[i]; end
    for i in (    6  ); y[i] = y[i-2] ⊕ y[i]; end
    for i in ( 3,5,7 ); y[i] = y[i-1] ⊕ y[i]; end
    y
end

#-

prefix8!([1:8;], +) == cumsum(1:8)

# In fact, this can generalize beyond just length-8 arrays:

## More magic
function prefix!(y, ⊕)
    l=length(y)
    k=ceil(Int, log2(l))
    @inbounds for j=1:k, i=2^j:2^j:min(l, 2^k)              #"reduce"
        y[i] = y[i-2^(j-1)] ⊕ y[i]
    end
    @inbounds for j=(k-1):-1:1, i=3*2^(j-1):2^j:min(l, 2^k) #"expand"
        y[i] = y[i-2^(j-1)] ⊕ y[i]
    end
    y
end

# -

A = rand(0:9, 123)
prefix!(copy(A), *) == cumprod(A)

# ## What is this magic?

# We can visualize the operations with a little bit of trickery. In Julia, arrays are simply types that expose the array protocol. In particular, they need to implement  methods for the generic functions `length`, `getindex` and `setindex!`. The last two are used in indexing operations, since statements
#
#     y = A[1]
#     A[3] = y
#
# get desugared to
#
#     y = getindex(A, 1)
#     setindex!(A, y, 3)
#
# respectively.
#
# We can trace through the iterable by introduce a dummy array type, `AccessArray`, which records every access to `getindex` and `setindex!`.
#
# Specifically:
#
# - `length(A::AccessArray)` returns the length of the array it wraps
# - `getindex(A::AccessArray, i)` records read access to the index `i` in the `A.read` field and then actually retuns the value in the array it wraps.
# - `setindex!(A::AccessArray, x, i)` records write access to the index `i`. The `A.history` field is appended with a new tuple consisting of the current `A.read` field and the index `i`, and then it performs the assignment.
#
# The way `AccessArray` works, it assumes an association between a single `setindex!` call and and all the preceding `getindex` calls since the previous `setindex!` call, which is sufficient for the purposes of tracing through prefix calls.

mutable struct AccessArray{T,N,A}
    data :: A
    read :: Vector{Int}
    history :: Vector{Tuple{Vector{Int},Int}}
end
AccessArray(A) = AccessArray{eltype(A), ndims(A), typeof(A)}(A, Int[], Int[])

Base.length(A::AccessArray) = length(A.data)

function Base.getindex(A::AccessArray, i::Int)
    push!(A.read, i)
    A.data[i]
end

function Base.setindex!(A::AccessArray, x, i::Int)
    push!(A.history, (A.read, i))
    A.read = Int[]
    A.data[i] = x
end

#-

M = AccessArray(rand(8))

#-

M[7] = M[3] + M[2]

#-

M.history


# So now we can trace the access pattern when calling `prefix8`!

A=prefix8!(AccessArray(rand(8)),+)

#-

A.history

# Now let's visualize this! Each entry in `A.history` is rendered by a gate object:

using Compose: circle, mm

#-

struct Gate{I,O}
    ins :: I
    outs :: O
end

import Gadfly.render

function render(G::Gate, x₁, y₁, y₀; rᵢ=0.1, rₒ=0.25)
    ipoints = [(i, y₀+rᵢ) for i in G.ins]
    opoints = [(i, y₀+0.5) for i in G.outs]
    igates  = [circle(i..., rᵢ) for i in ipoints]
    ogates  = [circle(i..., rₒ) for i in opoints]
    lines = [line([i, j]) for i in ipoints, j in opoints]
    compose(context(units=UnitBox(0.5,0,x₁,y₁+1)),
    compose(context(), stroke(colorant"black"), fill(colorant"white"),
            igates..., ogates...),
    compose(context(), linewidth(0.3mm), stroke(colorant"black"),
            lines...))
end

A=Gate([1,2],2)
render(A,2,0,0)

# Now we render the whole algorithm. We have to scan through the trace twice; the first time merely calculates the maximum depth that needs to be drawn and the second time actually generates the objects.

function render(A::AccessArray)
    #Scan to find maximum depth
    olast = depth = 0
    for y in A.history
        (any(y[1] .≤ olast)) && (depth += 1)
        olast = maximum(y[2])
    end
    maxdepth = depth

    olast = depth = 0
    C = []
    for y in A.history
        (any(y[1] .≤ olast)) && (depth += 1)
        push!(C, render(Gate(y...), length(A), maxdepth, depth))
        olast = maximum(y[2])
    end

    push!(C, compose(context(units=UnitBox(0.5,0,length(A),1)),
      [line([(i,0), (i,1)]) for i=1:length(A)]...,
    linewidth(0.1mm), stroke(colorant"grey")))
    compose(context(), C...)
end

#-

render(prefix!(AccessArray(zeros(8)), +))

# Now we can see that `prefix!` rearranges the operations to form two spanning trees:

render(prefix!(AccessArray(zeros(120)),+))

# as contrasted with the serial code:

render(prefix_serial!(AccessArray(zeros(8)),+))

#-

@manipulate for npp=1:180
    render(prefix!(AccessArray(zeros(npp)),+))
end

#-

@manipulate for npp=1:180
    render(prefix_serial!(AccessArray(zeros(npp)),+))
end

# # Now exploit the parallelism in the _algorithm_ to use a parallel _implementation_

using .Threads
function prefix_threads!(y, ⊕)
    l=length(y)
    k=ceil(Int, log2(l))
    for j=1:k
        @threads for i=2^j:2^j:min(l, 2^k)       #"reduce"
            @inbounds y[i] = y[i-2^(j-1)] ⊕ y[i]
        end
    end
    for j=(k-1):-1:1
        @threads for i=3*2^(j-1):2^j:min(l, 2^k) #"expand"
            @inbounds y[i] = y[i-2^(j-1)] ⊕ y[i]
        end
    end
    y
end

A = rand(500_000);

using BenchmarkTools
@btime prefix_serial!($(copy(A)), +);
@btime prefix!($(copy(A)), +);
@btime prefix_threads!($(copy(A)), +);

prefix_threads!(copy(A), +) == prefix!(copy(A), +) ≈ cumsum(A)

# # Thinking in parallel
#
# Notice how we didn't need to contort ourselves in making our algorithm
# work with `@threads`. We really did _just_ take a `@threads` on it and it
# just worked. It was both accurate _and_ fast.
#
# Coming up with rearrangements that make your particular algorithm parallel
# friendly isn't always easy, but when possible it makes everything else
# just fall out naturally.
#
# Finally, note that there can be clever ways to visualize algorithms as sanity checks.
