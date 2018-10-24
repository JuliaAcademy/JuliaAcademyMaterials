# # Course on Automatic Differentiation in Juia

# ## Why derivatives ?

# Derivatives are very useful in scietific computing.
# A common use case is in iterative routines which
# include finding roots and extreme points to nonlinear equations.
# Ther are ways to do nonlinear optimization and nonlinear equation solving
# without using derivatives but they are typically slow and non robust.
# By including "higher order information" or the derivative, the iterative routine
# gets more information and this opens up a whole new class of algorithms
# that can be orders of magnitude faster than the ones that do not use the derivative.
#
# Manually deriving the derivative of a function (or algorithm) can be
# tedious and error prone. A small error in the implementation of the derivative
# can lead to that the optimal order of convergence in an iterative method is lost.
# Indeed, the author of thise couse has lost many days from trying to locate that
# last bug in a derivative implementation.
#
# Thinking about the rules for derivatives, they are quite simpl (here using an apostrophe
# to denote the derivative):
#
# * Addition rule: $(f(x) + g(x))' = f'(x) + g'(x)$
# * Product rule: $(f(x)g(x))' = f(x)g'(x) + f'(x)g(x)$
# * Chain rule: $(f(g(x)))' = f'(g(x))g'(x)$
#
# It is natural to ask the question, should it not be possible to
# automate the computation of these derivatives
# and just have our program give back the derivative for us automatically.
# In most cases, the answer to this is yes, it is possible to get the exact derivative
# without having to implement any derivatives.
# This is known as Automatic Differentiation typically shortened as AD.
#
#
# ## Course goals
#
# In this course we will use the programming language Julia to look at AD.
# The course goals is that after successfully finishing the course, you
# should be able to:
# * Use some of the popular AD tools in Julia, both for forward mode and reverse mode AD.
# * Call optimization routines using derivatives computed with AD.
# * Understand the theoretical background behind dual numbers and how they can be used for forward mode AD.
# * Implement you own toy-version of forward mode AD.
