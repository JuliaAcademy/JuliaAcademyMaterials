using Test #src

# # Exercise 1
# 
# Set `x` to be a 20-item random sample with replacement from the integers 1 to 15.

x = "Change this String to be the random sample"

@assert length(x) == 20             # src
@assert all(x -> 1 ≤ x ≤ 15, x)     # src

# # Exercise 2
#
# What is the probability of 