# Basic control flow
#
# # Fibonnacci numbers
#
# The Fibonnacci sequence is: 1, 1, 2, 3, 5, 8, 13..., where each subsequent
# number is the sum of the prior two numbers. Create a function that computes
# the N-th Fibonnacci number.
#   Options:
#       * Iterative: use a for loop and keep track of prior two values to add them
#           What happens if you ask for the 95th number? The 96th? 97th?
#       * Recursive: work backwards and add the prior two numbers together
#           How long does it take to get the 40th number? The 41st? 42nd?
#       * Matrix magic



# # Monte-carlo pi
#
# Create a function that estimates pi by repeatedly throwing a dart at a square
# board. It's easy to test if the particular dart landed inside the circle —
# just ask if x^2 + y^2 < 1! Then comparing the number inside vs. the total
# gives you the ratio of the circle's area to the square's area.

using Images
load(download("http://corysimon.github.io/images/julia/myplot.png"))

# Plot the error as the number of iterations increases

# # Packages
#
# Add the Primes package. What does it do? How do you compute the 100th prime?
# A very large prime number?
#
#
