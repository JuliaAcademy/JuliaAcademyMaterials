import Pkg; Pkg.add(Pkg.PackageSpec(url="https://github.com/JuliaComputing/JuliaAcademyData.jl"))
using JuliaAcademyData; activate("Parallel_Computing")

# # SIMD: The parallelism that can (sometimes) happen automatically
#
# SIMD: Single-instruction, multiple data
#
# (Also confusingly called vectorization)

#-

# ## The architecture
#
# Instead of computing four sums sequentially:
#
# \begin{align}
# x_1 + y_1 &\rightarrow z_1 \\
# x_2 + y_2 &\rightarrow z_2 \\
# x_3 + y_3 &\rightarrow z_3 \\
# x_4 + y_4 &\rightarrow z_4
# \end{align}
#
# Modern processors have vector processing units that can do it all at once:
#
# $$
# \left(\begin{array}{cc}
# x_1 \\
# x_2 \\
# x_3 \\
# x_4
# \end{array}\right)
# +
# \left(\begin{array}{cc}
# y_1 \\
# y_2 \\
# y_3 \\
# y_4
# \end{array}\right)
# \rightarrow
# \left(\begin{array}{cc}
# z_1 \\
# z_2 \\
# z_3 \\
# z_4
# \end{array}\right)
# $$

#-

# ## Making it happen

#-

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
@btime simplesum($A)

# So, is that good?

@btime sum($A)

# We're slower that the builtin `sum` — and we're getting a different answer, too! Let's look at what happens with a 32-bit float instead of a 64 bit one. Each element has half the number of bits, so lets also double the length (so the total number of bits processed remains constant).

A32 = rand(Float32, length(A)*2)
@btime simplesum($A32)
@btime sum($A32);

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
# even if it would change the answer. Depending on your CPU, this may lead to 2x or 4x
# or even 8x parallelism. Essentially, Julia is computing independent sums for
# the even indices and the odd indices simultaneously:
#
# \begin{align}
# odds &\leftarrow 0 \\
# evens &\leftarrow 0 \\
# \text{loop}&\ \text{odd}\ i: \\
#     &\left(\begin{array}{cc}
# odds \\
# evens
# \end{array}\right)
# \leftarrow
# \left(\begin{array}{cc}
# odds \\
# evens
# \end{array}\right)
# +
# \left(\begin{array}{cc}
# x_{i} \\
# x_{i+1}
# \end{array}\right) \\
# total &\leftarrow evens + odds
# \end{align}
#
# In many cases, Julia can and does know that a for-loop can be SIMD-ed and it
# will take advantage of this by default!

B = rand(1:10, 100_000)
@btime simplesum($B)
@btime sum($B)
B32 = rand(Int32(1):Int32(10), length(B)*2)
@btime simplesum($B32)
@btime simdsum($B32)

# How can we see if something is getting vectorized?

@code_llvm simdsum(A32)

# So what are the challenges?
#
# * Biggest hurdle is that you have to convince Julia and LLVM that it's able to
#   use SIMD instructions for your given algorithm. That's not always possible.
# * There are lots of limitations of what can and cannot be SIMD-ed:

@doc @simd

# * You do need to think through the consequences of re-ordering your algorithm.

#-

# ## A slightly trickier case

using BenchmarkTools

#-

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
# with the very special `@simd ivdep` flag, but this can be disastrous:

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

#-

# ## SIMD
#
# * Exploits built-in parallelism in a processor
# * Best for small, tight innermost loops
# * Often happens automatically if you're careful
#     * Follow the [perforance best practices](https://docs.julialang.org/en/v1/manual/performance-tips/)
#     * `@inbounds` any array acesses
#     * No branches or (non-inlined) function calls
# * Can use `@simd` to allow Julia to break some rules to make it happen
#     * But be careful, especially with `@simd ivdep`!
# * Depending on processor and types involved, can yield 2-16x gains with extraordinarily little overhead
#     * Smaller datatypes can improve this further; use `Float32` instead of `Float64`
#       if possible, `Int32` instead of `Int64`, etc.
#     * When buying a new processor, look for [AVX-512](https://en.wikichip.org/wiki/x86/avx-512) support

