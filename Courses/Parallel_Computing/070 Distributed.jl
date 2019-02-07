# # Distributed

# Distributed (or multi-core or multi-process) parallelism

using Distributed
nprocs()
#-
addprocs()
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

@time @sync for i in 2:nprocs()
    @spawnat i work(100_000_000)
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
    for i in 2:nprocs()
        batch = 0:length(r)÷nworkers()-1
        futures[i-1] = @spawnat i partial_pi(batch .+ (i-2)*(length(r)÷nworkers()))
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

# ## Data movement

# Moving data is _expensive_! (from https://www.prowesscorp.com/computer-latency-at-a-human-scale/)
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

# But sometimes you need to see those intermediate values. If you have a
# very expensive computation relative to the communication overhead, there are
# several ways to do this. The easiest is `pmap`:

@time pmap(partial_pi, [(0:9999) .+ offset for offset in 0:10000:r[end]-1])

# But if we have a large computation relative to the number of return values,
# pmap is great and easy.
#
# Increase the work on each worker by 100x and reduce the amount of communication by 100x:
@time pmap(partial_pi, [(0:999999) .+ offset for offset in 0:1000000:r[end]-1])

# There are other ways of doing this, though, too.

# ### The SharedArray

# ### The DistributedArray

# ## Code movement

# Each node is _completely_ independent; it's like starting brand new, separate
# Julia processes yourself. By default, `addprocs()` just launches the
# appropriate number of workers for the current workstation that you're on, but
# you can easily connect them to remote machines via SSH or even through cluster
# managers.

# Those `@everywhere`s above are very important! They run the given expression
# on all workers to make sure the state between them is consistent. Without it,
# you'll see errors like this:

hello() = "hello world"

r = @spawnat 2 hello()
fetch(r)

# Note that this applies to packages, too!

using Statistics # The Statistics stdlib defines mean
fetch(@spawnat 2 mean(rand(100_000)))

#-

@everywhere using Statistics
fetch(@spawnat 2 mean(rand(100_000)))

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
#     * `SharedArray`s can be an easier drop-in replacement of some complicated data transfers, but at a (time) cost
#     * `DistributedArray`s can be tricky to work with but can be very efficient
#
