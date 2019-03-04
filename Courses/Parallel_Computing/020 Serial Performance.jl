# # Fast (serial) programming with Julia
#
# Yes, this is a parallel computing course — but to write efficient parallel
# programs we first must learn how to write fast serial Julia code. This is
# a rapid primer in high performance (serial) programming.
#
# I _highly_ recommend reviewing the [Performance Tips](https://docs.julialang.org/en/v1.1/manual/performance-tips/)
# in the manual. This is only going to briefly introduce some of the main concepts.

#-

# ## Measure, measure, measure.
#
# It is very easy to experiment in Julia; you can rapidly try many options and
# see what is the fastest.

#-

# Use the [BenchmarkTools](https://github.com/JuliaCI/BenchmarkTools.jl) package:

using BenchmarkTools

"""
    findclosest(data, point)

A simple example that returns the element in `data` that is closest to `point`
"""
function findclosest(data, point)
    _, index =  findmin(abs.(data .- point))
    return data[index]
end
data = rand(5000)
findclosest(data, 0.5)

#-

@time findclosest(data, 0.5)

#-

@benchmark findclosest($data, $0.5)

# ### Profile!

using Profile

Profile.clear()
@profile for _ in 1:100000; findclosest(data, 0.5); end

Profile.print(maxdepth=11)

# ### Iterate!
#
# Before we had:
# ```julia
# function findclosest(data, point)
#     _, index =  findmin(abs.(data .- point))
#     return data[index]
# end
# ```
#
# Let's come up with a new definition that can combine the two operations:

function findclosest2(data, point)
    bestval = first(data)
    bestdist = abs(bestval - point)
    for elt in data
        dist = abs(elt - point)
        if dist < bestdist
            bestval = elt
            bestdist = dist
        end
    end
    return bestval
end

## And do a spot-check to make sure we did the optimization correctly:
findclosest2(data, 0.5) == findclosest(data, .5)

#-

@benchmark findclosest2($data, $0.5)

# ## A quick word on macros
#
# Macros are those funny things starting with `@`. They can reinterpret what
# you write and do something different — essentially introducing a new keyword.
#
# For example, the `@assert` macro simply takes an expression and throws an
# exception if it returns `false`.

@assert 2+2 == 4

# It does this by literally re-writing what you wrote. You can see it in action
# with `@macroexpand`

@macroexpand @assert 2+2 == 4

# Each macro can define its own special syntax, and this is used extensively for
# code introspection, serial performance improvements, and — perhaps most
# importantly — parallelization perimitives!

#-

# ## How is Julia fast?
#
# By understanding the basics of how Julia _can_ be fast, you can get a better
# sense for how to write fast Julia code.
#
# Perhaps most importantly, Julia can reason about types. Recall: this is the definition of `findclosest2`:
#
# ```julia
# function findclosest2(data, point)
#     bestval = first(data)
#     bestdist = abs(bestval - point)
#     for elt in data
#         dist = abs(elt - point)
#         if dist < bestdist
#             bestval = elt
#             bestdist = dist
#         end
#     end
#     return bestval
# end
# ```

@code_typed optimize=false findclosest2(data, 0.5)

#-

typeof(data)

#-

newdata = Real[data...]
typeof(newdata)

#-

@code_typed optimize=false findclosest2(newdata, 0.5)

#-

@benchmark findclosest2(newdata, 0.5)

#-

@code_warntype findclosest2(newdata, 0.5)

# ### Type stability
#
# A function is called type-stable if Julia is able to infer what the output
# type will be based purely on the types of the inputs.
#
# Things that thwart type stability:
# * Running things in global scope: create functions instead!
# * Non-concretely typed containers
# * Structs with abstractly-typed fields
# * Non-constant globals (they might change!)
# * Functions that change what they return based on the _values_:

#-

# #### More on macros
#
# Each and every macro can define its own syntax. The `@benchmark` macro uses `$` in a special way.
# The goal behind `@benchmark` is to evaluate the performance of a code snippet
# as though it were written in a function. Use `$` to flag what will be an argument
# or local variable in the function. Forgetting to use `$`s may result in faster
# or slower timings than real-world performance.

x = 0.5 # non-constant global
@btime sin(x)
@btime sin($x)

#-

@btime sin(0.5) # constant literal!
@btime sin($0.5)

# ### Specializations
#
# Julia's reasoning about types is particularly important since it generates
# specialized machine code specifically for the given arguments.

@code_llvm 1 + 2

# This applies just the same to any functions we write — even the more complicated ones:

@code_llvm findclosest2(Float32[2.2,3.4,4.5],Float32(3.2))

# This applies just the same to any functions we write — even the more complicated ones:

remove_comments(s) = join(filter(x->!startswith(x, ";"), split(s, "\n")), "\n")
sprint(code_llvm, findclosest2, Tuple{Vector{Float32}, Int}) |> remove_comments |> print

# ## Modern hardware effects
#
# There are lots of little performance quirks in modern computers; I'll just
# cover two interesting ones here:

@benchmark findclosest2($data, $0.5)

#-

sorteddata = sort(data)
@benchmark findclosest2($sorteddata, $0.5)

# Unfortunately, this isn't demonstrable on a hardened cloud platform... because
# it's a huge security risk!
#
# * https://meltdownattack.com
# * https://discourse.julialang.org/t/psa-microbenchmarks-remember-branch-history/17436

idxs = sortperm(data)
sortedview = @view data[idxs]
@benchmark findclosest2($sortedview, $0.5)

# ### Memory latencies
#
# | System Event                   | Actual Latency | Scaled Latency |
# | ------------------------------ | -------------- | -------------- |
# | One CPU cycle                  |     0.4 ns     |     1 s        |
# | Level 1 cache access           |     0.9 ns     |     2 s        |
# | Level 2 cache access           |     2.8 ns     |     7 s        |
# | Level 3 cache access           |      28 ns     |     1 min      |
# | Main memory access (DDR DIMM)  |    ~100 ns     |     4 min      |
# | Intel Optane memory access     |     <10 μs     |     7 hrs      |
# | NVMe SSD I/O                   |     ~25 μs     |    17 hrs      |
# | SSD I/O                        |  50–150 μs     | 1.5–4 days     |
# | Rotational disk I/O            |    1–10 ms     |   1–9 months   |
# | Internet call: SF to NYC       |      65 ms     |     5 years    |
# | Internet call: SF to Hong Kong |     141 ms     |    11 years    |
#
#  (from https://www.prowesscorp.com/computer-latency-at-a-human-scale/)

#-

# # Key Takeaways
#
# * Measure, measure, measure!
# * Get familiar with the [Performance Tips](https://docs.julialang.org/en/v1/manual/performance-tips/)
# * Don't be scared of `@code_typed`/`@code_warntype` and `@code_llvm`

