# # SIMD: The parallelism that can (sometimes) happen automatically
#
# SIMD: Single-instruction, multiple data
#
# (Also confusingly called vectorization)

# ## The architechture

# ## Making it happen

# Simple task: compute the sum of a vector:

A = rand(100_000)
function simplesum(A)
    result = zero(eltype(A))
    for i in eachindex(A)
        @inbounds result += A[i]
    end
    return result
end

simplesum(A)

#-

using BenchmarkTools
@btime mysum($A)

# So, is that good?

@btime sum($A)

# Whoa, we're a lot slower that the builtin `sum` — and we're getting a different answer, too!

A32 = rand(Float32, length(A)*2)
@btime simplesum($A32)
@btime sum($A32)

# That's even worse! What's going on here?  We're seeing an even multiple number
# difference in our performance — perhaps Julia's builtin sum is using some
# parallelism? Let's try using SIMD ourselves:

function simdsum(A)
    result = zero(eltype(A))
    @simd for i in eachindex(A)
        @inbounds result += A[i]
    end
    return result
end
@btime simdsum($A)
@btime simdsum($A32)

# What did that do and why don't we always use `@simd for` — or why doesn't Julia
# just always use `@simd` for every `for` loop automatically?  Look at the values:

simplesum(A), simdsum(A), sum(A)

#-

simplesum(A32), simdsum(A32), sum(A32)

# Why aren't they the same?
#
# Without `@simd`, Julia is doing _exactly_ what we told it to do: it's taking
# each element of our array and adding it to a big pile sequentially. Our answer
# is smaller than what Julia's builtin `sum` thinks it is: that's because as our
# pile gets bigger we begin losing the lower bits of each element that we're
# adding, and those small losses begin to add up!
#
# The `@simd` macro tells Julia that it can re-arrange floating point additions —
# even if it would change the answer.
#
# In many cases, Julia can and does know that a for-loop can be SIMD-ed and it
# will take advantage of this by default!

B = rand(1:10, 100_000)
@btime simplesum($B)
@btime sum($B)
B32 = rand(Int32(1):Int32(10), length(B)*2)
@btime simplesum($B)

# How can we see if something is getting vectorized?

@code_llvm simdsum(A32)

# So what are the challenges?
#
# * Biggest hurdle is that you have to convince Julia and LLVM that it's able to
#   use SIMD instructions for your given algorithm. That's not always possible.
# * There are lots of limitations of what can and cannot be SIMD-ed:
@doc @simd
# * You do need to think through the consequences of re-ordering your algorithm.

# ## A slightly trickier case

function diff!(A, B)
    A[1] = B[1]
    for i in 2:length(A)
        @inbounds A[i] = B[i] - B[i-1]
    end
    return A
end
A = zeros(Float32, 100_000)
B = rand(Float32, 100_000)

diff!(A, B)
[B[1];diff(B)] == A

#-
@btime diff!($A, $B)
@btime diff($B);

# But what happens if we do it in-place?
Bcopy = copy(B)
@btime diff!($Bcopy, $Bcopy);

# What happened?

@code_llvm diff!(A, B)

# We can manually assert that arrays don't alias (or have any loop-dependencies),
# but this can be disastrous:
function unsafe_diff!(A, B)
    A[1] = B[1]
    @simd ivdep for i in 2:length(A)
        @inbounds A[i] = B[i] - B[i-1]
    end
    return A
end
@btime unsafe_diff!($A, $B)
[B[1];diff(B)] == A
Bcopy = copy(B)
unsafe_diff!(Bcopy, Bcopy)
[B[1];diff(B)] == Bcopy


# If you really want to get your hands dirty, you can use the [SIMD.jl](https://github.com/eschnett/SIMD.jl)
# package to manually specify those `<8 x float>` things that LLVM generates.
# BUT: this is tricky and a pain; often it's just to be aware of what makes
# Julia code automatically SIMD-able, some of the cases where it may fail, and
# how to check its work.
