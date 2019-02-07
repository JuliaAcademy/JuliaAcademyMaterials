# # Parallel Algorithms: Thinking in Parallel

function prefix_serial!(y, ⊕)
    for i=2:length(y)
        y[i] = y[i-1] ⊕ y[i]
        end
    y
end

function prefix!(y, ⊕)
    l=length(y)
    k=ceil(Int, log2(l))
    @inbounds for j=1:k, i=2^j:2^j:min(l, 2^k)              #"reduce"
        y[i] = y[i-2^(j-1)] ⊕ y[i]
    end
    @inbounds for j=(k-1):-1:1, i=3*2^(j-1):2^j:min(l, 2^k) #"broadcast"
        y[i] = y[i-2^(j-1)] ⊕ y[i]
    end
    y
end

using .Threads
function prefix_threads!(y, ⊕)
    l=length(y)
    k=ceil(Int, log2(l))
    for j=1:k
        @threads for i=2^j:2^j:min(l, 2^k)              #"reduce"
            @inbounds y[i] = y[i-2^(j-1)] ⊕ y[i]
        end
    end
    for j=(k-1):-1:1
        @threads for i=3*2^(j-1):2^j:min(l, 2^k) #"broadcast"
            @inbounds y[i] = y[i-2^(j-1)] ⊕ y[i]
        end
    end
    y
end

ylocal = rand(500_000);

using BenchmarkTools
@btime prefix_serial!($(copy(ylocal)), +);
@btime prefix!($(copy(ylocal)), +);
@btime prefix_threads!($(copy(ylocal)), +);

prefix_threads!(copy(ylocal), +) == prefix!(copy(ylocal), +) ≈ cumsum(ylocal)
