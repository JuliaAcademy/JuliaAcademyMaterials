# # Distributed (or multi-core or multi-process) parallelism
#
# Julia has a built-in standard library — Distributed — that allows you to
# start and run multiple concurrent Julia processes. Imagine starting a slew
# of Julia instances and then having an easy way to run code on each and every
# one of them; that's what Distributed provides.
#
# ![](images/Julia6x.png)

using Distributed
nprocs()

#-

addprocs(4)
@sync @everywhere workers() include("/opt/julia-1.0/etc/julia/startup.jl") # Needed just for JuliaBox
nprocs()

#-

myid()

# Now we can easily communicate with the other nodes:

r = @spawnat 2 (myid(), rand())

#-

fetch(r)

# This works kinda like an `@async` task!

@time r = @spawnat 2 (sleep(1), rand())
@time fetch(r)

# So we can repeat the same examples from tasks:

@time for i in 2:nprocs() # proc 1 is the controlling node
    @spawnat i sleep(1)
end

#-

@time @sync for i in 2:nprocs()
    @spawnat i sleep(1)
end

# Except unlike tasks, we're executing the code on a separate process — which
# can be performed on a different processor in parallel!

@everywhere function work(N)
    series = 1.0
    for i in 1:N
        series += (isodd(i) ? -1 : 1) / (i*2+1)
    end
    return 4*series
end

#-

@time work(1_000_000_000)
@time @sync for i in workers()
    @spawnat i work(1_000_000_000)
end

# Of course, this isn't very helpful. We're just performing exactly the same
# calculation on every worker... and then completely ignoring the result! Let's
# restructure our computation to be a bit more parallel friendly:

@everywhere function partial_pi(r)
    series = 0.0
    for i in r
        series += (isodd(i) ? -1 : 1) / (i*2+1)
    end
    return 4*series
end
a = partial_pi(0:999)
a, a-pi

#-

b = partial_pi(1000:9999)
(a + b), (a+b) - pi

# So now we can distribute this computation across our many workers!

r = 0:10_000_000_000
futures = Array{Future}(undef, nworkers())
@time begin
    for (i, id) in enumerate(workers())
        batch = 0:length(r)÷nworkers()-1
        futures[i] = @spawnat id partial_pi(batch .+ (i-1)*(length(r)÷nworkers()))
    end
    p = sum(fetch.(futures))
end
p - pi

# But that's rather annoying — needing to carefully divide up our workflow and
# manually collect all our results and such.  There's an easier way:

@time p = @distributed (+) for r in [(0:9999) .+ offset for offset in 0:10000:r[end]-1]
    partial_pi(r)
end
p - pi

# Why is this different from `@threads for` and `@simd for`? Why not just
# `@distributed for`?  Why the `@distributed (+) for`?

#-

# ## Data movement

#-

# Remember: Moving data is _expensive_!
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
# You really don't want to be taking a trip to the moon very frequently.
# Communication between processes can indeed be as expensive as hitting a disk —
# sometimes they're even implemented that way.
#
# So that's why Julia has special support for reductions built in to the
# `@distributed` macro: each worker can do its own (intermediate) reduction
# before returning just one value to our master node.

#-

# But sometimes you need to see those intermediate values. If you have a
# very expensive computation relative to the communication overhead, there are
# several ways to do this. The easiest is `pmap`:

@time pmap(partial_pi, [(0:99999) .+ offset for offset in 0:100000:r[end]-1])

# But if we have a large computation relative to the number of return values,
# pmap is great and easy.
#
# Increase the work on each worker by 100x and reduce the amount of communication by 100x:

@time pmap(partial_pi, [(0:9999999) .+ offset for offset in 0:10000000:r[end]-1])

# There are other ways of doing this, though, too — we'll get to them in a minute.
# But first, there's something else that I glossed over: the `@everywhere`s above.

#-

# ## Code movement

#-

# Each node is _completely_ independent; it's like starting brand new, separate
# Julia processes yourself. By default, `addprocs()` just launches the
# appropriate number of workers for the current workstation that you're on, but
# you can easily connect them to remote machines via SSH or even through cluster
# managers.

#-

# Those `@everywhere`s above are very important! They run the given expression
# on all workers to make sure the state between them is consistent. Without it,
# you'll see errors like this:

hello() = "hello world"
r = @spawnat 2 hello()

#-

fetch(r)

# Note that this applies to packages, too!

using Statistics # The Statistics stdlib defines mean
fetch(@spawnat 2 mean(rand(100_000)))

#-

@everywhere using Statistics
fetch(@spawnat 2 mean(rand(100_000)))

# # Other ways to structure and/or share data between processes
#
# Unlike `@threads`, we no longer have access to the same memory. While this
# does make expressing some algorithms a little more tricky, the "default"
# is much safer! There isn't any shared state to begin with, so it's harder
# to write an incorrect algorithm. It's also just harder to write some
# algorithms in the first place.
#
# So there are some special array types that can help bridge the gap between
# processes and make writing parallel code a bit easier.

#-

# ## The `SharedArray`
#
# If all workers are on the same physical machine, while they cannot share
# memory, they do all have shared access to the same hard drive(s)!
#
# The `SharedArray` makes use of this fact, allowing concurrent accesses to the
# same array — somewhat akin to threads default state.
#
# This is the prefix definition from the "thinking in parallel" course:
#
# ```
# using .Threads
# function prefix_threads!(y, ⊕)
#     l=length(y)
#     k=ceil(Int, log2(l))
#     for j=1:k
#         @threads for i=2^j:2^j:min(l, 2^k)       #"reduce"
#             @inbounds y[i] = y[i-2^(j-1)] ⊕ y[i]
#         end
#     end
#     for j=(k-1):-1:1
#         @threads for i=3*2^(j-1):2^j:min(l, 2^k) #"expand"
#             @inbounds y[i] = y[i-2^(j-1)] ⊕ y[i]
#         end
#     end
#     y
# end
# ```

using SharedArrays
function prefix!(y::SharedArray, ⊕)
    l=length(y)
    k=ceil(Int, log2(l))
    for j=1:k
        @distributed for i=2^j:2^j:min(l, 2^k)       #"reduce"
            @inbounds y[i] = y[i-2^(j-1)] ⊕ y[i]
        end
    end
    for j=(k-1):-1:1
        @distributed for i=3*2^(j-1):2^j:min(l, 2^k) #"expand"
            @inbounds y[i] = y[i-2^(j-1)] ⊕ y[i]
        end
    end
    y
end
data = rand(1_000_000);
A = SharedArray(data);

#-

prefix!(SharedArray(data), +) # compile
@time prefix!(A, +);

#-

A ≈ cumsum(data)

# What went wrong?

function prefix!(y::SharedArray, ⊕)
    l=length(y)
    k=ceil(Int, log2(l))
    for j=1:k
        @sync @distributed for i=2^j:2^j:min(l, 2^k)       #"reduce"
            @inbounds y[i] = y[i-2^(j-1)] ⊕ y[i]
        end
    end
    for j=(k-1):-1:1
        @sync @distributed for i=3*2^(j-1):2^j:min(l, 2^k) #"expand"
            @inbounds y[i] = y[i-2^(j-1)] ⊕ y[i]
        end
    end
    y
end
A = SharedArray(data)
@time prefix!(A, +)

#-

A ≈ cumsum(data)

# ## DistributedArrays
#
# We can, though, turn the problem on its head and allow the _data_ itself
# to determine how the problem gets split up. This can save us tons of indexing
# headaches.

@everywhere using Distributed
@everywhere using DistributedArrays
A = DArray(I->fill(myid(), length.(I)), (24, 24))

# The first argument takes a function that transforms the given set of indices
# to the _local portion_ of the distributed array.

A = DArray((24,24)) do I
    @show I
    fill(myid(), length.(I))
end

# Notice that none of the array actually lives on processor 1, but we can still
# display the contents — when we do we're requesting all workers give us their
# current data! While we've only talked about master-worker communcation so far,
# workers can communicate directly amongst themselves, too (by default).

@everywhere using BenchmarkTools
fetch(@spawnat 2 @benchmark $A[1,1])

#-

fetch(@spawnat 2 @benchmark $A[end,end])

# So it's fastest to work on a `DArray`'s "local" portion, but it's _possible_
# to grab other data if need be. This is perfect for any sort of tiled operation
# that works on neighboring values (like image filtering/convolution). Or Conway's
# game of life!

function life_step(d::DArray)
    DArray(size(d),procs(d)) do I
        # Compute the indices of the outside edge (that will come from other processors)
        top   = mod1(first(I[1])-1,size(d,1))
        bot   = mod1( last(I[1])+1,size(d,1))
        left  = mod1(first(I[2])-1,size(d,2))
        right = mod1( last(I[2])+1,size(d,2))
        # Create a new, temporary array that holds the local part + outside edge
        old = Array{Bool}(undef, length(I[1])+2, length(I[2])+2)
        # These accesses will pull data from other processors
        old[1      , 1      ] = d[top , left]
        old[2:end-1, 1      ] = d[I[1], left]   # left side (and corners)
        old[end    , 1      ] = d[bot , left]
        old[1      , end    ] = d[top , right]
        old[2:end-1, end    ] = d[I[1], right]  # right side (and corners)
        old[end    , end    ] = d[bot , right]
        old[1      , 2:end-1] = d[top , I[2]]   # top
        old[end    , 2:end-1] = d[bot , I[2]]   # bottom
        # But this big one is all local!
        old[2:end-1, 2:end-1] = d[I[1], I[2]]   # middle

        life_rule(old) # Compute the new segment!
    end
end
@everywhere function life_rule(old)
    # Now this part — the computational part — is entirely local and on Arrays!
    m, n = size(old)
    new = similar(old, m-2, n-2)
    for j = 2:n-1
        @inbounds for i = 2:m-1
            nc = (+)(old[i-1,j-1], old[i-1,j], old[i-1,j+1],
                     old[i  ,j-1],             old[i  ,j+1],
                     old[i+1,j-1], old[i+1,j], old[i+1,j+1])
            new[i-1,j-1] = (nc == 3 || nc == 2 && old[i,j])
        end
    end
    new
end

#-

A = DArray(I->rand(Bool, length.(I)), (20,20))
using Colors
Gray.(A)

#-

B = copy(A)

#-

B = Gray.(life_step(B))

# ## Clusters and more ways to distribute
#
# You can easily connect to completely separate machines with SSH access built in!
# But there are many other ways to connect to clusters:
#
# * [JuliaRun](https://juliacomputing.com/products/juliarun)
# * [Kubernetes](https://juliacomputing.com/blog/2018/12/15/kuber.html)
# * [MPI](https://github.com/JuliaParallel/MPI.jl)
# * [Cluster job queues with ClusterManagers](https://github.com/JuliaParallel/ClusterManagers.jl)
# * [Hadoop](https://github.com/JuliaParallel/Elly.jl)
# * [Spark](https://github.com/dfdx/Spark.jl)

#-

# # Multi-process parallelism is the heavy-duty workhorse in Julia
#
# It can tackle very large problems and distribute across a very large number
# of workers. Key things to remember
#
# * Each worker is a completely independent Julia process
#     * Data must move to them
#     * Code must move to them
# * Structure your algorithms and use a distributed mechanism that fits with the
#   time and memory parameters of your problem
#     * `@distributed` can be good for reductions and even relatively fast inner loops with limited (or no) explicit data transfer
#     * `pmap` is great for very expensive inner loops that return a value
#     * `SharedArray`s can be an easier drop-in replacement for threading-like behaviors (on a single machine)
#     * `DistributedArray`s can turn the problem on its head and let the data do the work splitting!
