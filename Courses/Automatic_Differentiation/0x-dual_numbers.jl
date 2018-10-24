

###

struct Dual{T}
    x::T
    eps::T
end





# Define enough functions on the `Dual` number so you can differentiate:

f(x) = (sin(x) * exp(x)) / x^4


