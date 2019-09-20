# # Iterators

# `for` loops "lower" to `while` loops plus calls to the `iterate` function:

# ```
# for i in iter
#     # body
# end
# ```

# is internally converted to:

# ```
# next = iterate(iter)
# while next !== nothing
#     (i, state) = next
#     # body
#     next = iterate(iter, state)
# end
# ```

# The same applies to comprehensions and generators.

# Note `nothing` is a singleton value (the only value of its type `Nothing`)
# used by convention when there is no value to return.

typeof(print("hello"))

A = ['a','b','c'];

iterate(A)
iterate(A, 2)
iterate(A, 3)
iterate(A, 4)

# Iteration is also used by "destructuring" assignment:

x, y = A
x
y

# Yet another user of this "iteration protocol" is argument "splatting":

string(A)
string('a','b','c')
string(A...)

# ## Iteration utilities

# `collect` gives you elements of an iterator as an array.
# Comprehensions work by calling `collect` on a generator.

collect(pairs(A))
summary(collect(pairs(A)))
collect(zip(100:102,A))

# There are several other highly popular iteration utilities in the
# built-in module `Iterators`:
# - `enumerate`
# - `rest`
# - `take`
# - `drop`
# - `product`
# - `flatten`
# - `partition`

#-

# Some iterators are infinite!
# - `countfrom`
# - `repeated`
# - `cycle`

I = zip(Iterators.cycle(0:1), Iterators.flatten([[2,3],[4,5]]))

collect(I)

collect(Iterators.product(I,A))

string(I...)

# ## Generator syntax

# Generators provide convenient syntax for composing map, filter, product, and
# flatten.

collect( (2i,j,k) for i = 1:3, j = 1:3 for k = 1:j if i==k)

change(x) = ( (p,n÷5,d÷10,q÷25) for q = 0:25:x for d = 0:10:x-q for n = 0:5:x-q-d for p = x-q-d-n )

collect(change(200))

# ## Defining iterators
#
# Defining iterable types is not necessary that often, since you can often
# get what you need by composing existing iterators.

struct SimpleRange
    lo::Int
    hi::Int
end

Base.iterate(r::SimpleRange, state = r.lo) = state > r.hi ? nothing : (state, state+1)
Base.length(r::SimpleRange) = r.hi - r.lo + 1

collect(SimpleRange(2,8))

# ## Iterator traits
#
# For many algorithms, it's useful to know certain properties of an iterator
# up front.
# The most useful is whether it has a known length.

Base.IteratorSize([1])
Base.IteratorSize(Iterators.repeated(1))
Base.IteratorSize(eachline(open("/dev/null")))

# ## Exercise
#
# Define an iterator giving the first N fibonacci numbers.
