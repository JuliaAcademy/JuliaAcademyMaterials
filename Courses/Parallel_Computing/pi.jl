import Pkg; Pkg.activate(@__DIR__)
using Plots
using CuArrays, CuArrays.CURAND
using FillArrays
using .Threads
using Random, Future
using Distributed

# # Basic task: try to speed up findpi with some sort of parallelism.

function findpi(n)
    inside = 0
    for i in 1:n
        x, y = rand(), rand()
        if x^2 + y^2 <= 1
            inside +=1
        end
    end
    return 4 * inside / n
end
xs = round.(Int, 10.0.^(0:.5:8.5))
@timed findpi(1)
ts = [(@timed findpi(x))[2] for x in xs]
plot(log10.(xs), log10.(ts), label="linear", legend=:topleft, xlabel="log10 iterations", ylabel="log10 seconds")

# # Threads
# Let's do the naive and obvious thing: throw @threads on the loop
# (after starting Julia up with 8 threads on a 12 core system)

function findpi_threads(n)
    inside = 0
    @threads for i in 1:n
        x, y = rand(), rand()
        if x^2 + y^2 <= 1
            inside += 1
        end
    end
    return 4 * inside / n
end
@timed findpi_threads(1)
#ts_threads = [(@timed findpi_threads(x))[2] for x in xs]
#plot!(log10.(xs), log10.(ts_threads), label="threads")

# That gave the wrong answer! Of course, all threads are reading/writing to
# the same variable — causing a race condition. And we're still slower!
# Let's use atomics:

function findpi_threads_atomic(n)
    inside = Threads.Atomic{Int}(0)
    @threads for i in 1:n
        x, y = rand(), rand()
        if x^2 + y^2 <= 1
            Threads.atomic_add!(inside, 1)
        end
    end
    return 4 * inside[] / n
end
@timed findpi_threads_atomic(1)

# ts_threads_atomic = [(@timed findpi_threads_atomic(x))[2] for x in xs]
# plot!(log10.(xs), log10.(ts_threads_atomic), label="atomic threads") # skip this plot — it's not really different

#-

# Still slower, and while the answer is closer it's still off! Ah, the RNG uses
# and mutates global state, causing some threads to repeat the same random
# sequence. Maybe try dividing the sum to thread-unique array slots?

function findpi_threads_divided(n)
    inside = zeros(Int, nthreads())
    @threads for i in 1:n
        x, y = rand(), rand()
        if x^2 + y^2 <= 1
            @inbounds inside[threadid()] += 1
        end
    end
    return 4 * sum(inside) / n
end
@timed findpi_threads_divided(1)

# ts_threads_divided = [(@timed findpi_threads_divided(x))[2] for x in xs]
# plot!(log10.(xs), log10.(ts_threads_divided), label="divided threads") # skip

#-

# Now let's fix that slight error by using independent RNG streams for each thread:

const rngs = let m = MersenneTwister(1)
        [m; accumulate(Future.randjump, fill(big(10)^20, nthreads()-1), init=m)]
    end;
function findpi_threads_divided_safe(n)
    inside = zeros(Int, nthreads())

    @threads for i in 1:n
        @inbounds x, y = rand(rngs[threadid()]), rand(rngs[threadid()])
        if x^2 + y^2 <= 1
            inside[threadid()] += 1
        end
    end
    return 4 * sum(inside) / n
end
@timed findpi_threads_divided_safe(1)

# ts_threads_divided_safe = [(@timed findpi_threads_divided_safe(x))[2] for x in xs]
# plot!(log10.(xs), log10.(ts_threads_divided_safe), label="divided safe threads")

#-

# Fixed the numerical error, but still slower! What about pre-computing our
# entire sequence of random numbers up front?

function findpi_threads_prealloc_divided(n)
    inside = zeros(Int, nthreads())
    rands = rand(2, n)
    @threads for i in 1:n
        @inbounds x, y = rands[1, i], rands[2, i]
        if x^2 + y^2 <= 1
            @inbounds inside[threadid()] += 1
        end
    end
    return 4 * sum(inside) / n
end
@timed findpi_threads_prealloc_divided(1)

# ts_threads_prealloc_divided = [(@timed findpi_threads_prealloc_divided(x))[2] for x in xs]
# plot!(log10.(xs), log10.(ts_threads_prealloc_divided), label="pre-alloc divided threads")

#-

# Try putting the RNGs on the threads:

const ThreadRNG = Vector{MersenneTwister}(undef, nthreads())
@noinline function init_thread_rng()
    # Allocate the random number generator on the thread's own heap lazily
    # instead of the master thread heap to minimize memory conflict.
    ThreadRNG[Threads.threadid()] = MersenneTwister(Threads.threadid())
end
@threads for i in 1:nthreads()
    init_thread_rng()
end
function findpi_threads_divided_safer(n)
    inside = zeros(Int, nthreads())

    @threads for i in 1:nthreads()
        rng = ThreadRNG[threadid()]
        v = 0
        for j in 1:n÷nthreads()
            @inbounds x, y = rand(rng), rand(rng)
            if x^2 + y^2 <= 1
                v += 1
            end
        end
        inside[threadid()] = v
    end
    return 4 * sum(inside) / n
end
@timed findpi_threads_divided_safer(1)
ts_threads_divided_safer = [(@timed findpi_threads_divided_safer(x))[2] for x in xs]
plot!(log10.(xs), log10.(ts_threads_divided_safer), label="divided safer threads")

# Nope, still slower. Does SIMD/fastmath gain us anything with rand?

function findpi_simd(n)
    inside = 0
    @fastmath @simd for i in 1:n
        x, y = rand(), rand()
        inside += Int(x^2 + y^2 <= 1)
    end
    return 4 * inside / n
end
@timed findpi_simd(1)

# ts_simd = [(@timed findpi_simd(x))[2] for x in xs]
# plot!(log10.(xs), log10.(ts_simd), label="simd")

function findpi_simd_prealloc(n)
    inside = 0
    rands = rand(2, n)
    @fastmath @simd for i in 1:n
        @inbounds x, y = rands[1, i], rands[2, i]
        inside += Int(x^2 + y^2 <= 1)
    end
    return 4 * inside / n
end
@timed findpi_simd_prealloc(1)

# ts_simd_prealloc = [(@timed findpi_simd_prealloc(x))[2] for x in xs]
# plot!(log10.(xs), log10.(ts_simd_prealloc), label="pre-alloc simd")

#-

# Nope, no gains there, either (didn't really expect any). Let's use Distributed

using Distributed
addprocs(11 - nprocs())

function findpi_distributed(n)
    inside = @distributed (+) for i in 1:n
        x, y = rand(), rand()
        Int(x^2 + y^2 <= 1)
    end
    return 4 * inside / n
end
@timed findpi_distributed(1)
ts_distributed = [(@timed findpi_distributed(x))[2] for x in xs]
plot!(log10.(xs), log10.(ts_distributed), label="distributed")

# Finally! An ~8x improvement with 8 workers.

function findpi_gpu(n)
    4 * sum(curand(Float64, n).^2 .+ curand(Float64, n).^2 .<= 1) / n
end
@timed findpi_gpu(1)
ts_gpu = [(@timed findpi_gpu(x))[2] for x in xs]
plot!(log10.(xs), log10.(ts_gpu), label="gpu")

# Even better!

#-

# Use a phony FillArray to define the broadcast shape (total number of elements)

function findpi_gpu_broadcast(n)
    4 * sum(Fill(0.0, n) .+ curand(Float64).^2 .+ curand(Float64).^2 .<= 1.0) / n
end
@timed findpi_gpu_broadcast(1)
ts_gpu_broadcast = [(@timed findpi_gpu_broadcast(x))[2] for x in xs]
plot!(log10.(xs), log10.(ts_gpu_broadcast), label="gpu lazy broadcast")

savefig("times.svg")
