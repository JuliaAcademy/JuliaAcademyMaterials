# # Multithreading
#
using .Threads

nthreads()

threadid()

A = zeros(Int, nthreads())
for i in 1:nthreads()
    A[threadid()] = threadid()
end
A

#-

@threads for i in 1:nthreads()
    A[threadid()] = threadid()
end
A
