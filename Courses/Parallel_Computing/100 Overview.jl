# # Summary
#
# * Challenges of parallel computing
#     * Order of execution
#         * execution of out order of Possibility
#         * simultaneous access and mutation
#     * Data access and movement
#     * Code access and movement
#     * Appropriately matching the parallelism strategy to your machine capabilities
#     * Appropriately matching the parallelism strategy with the problem at hand
#
# * Parallelism strategies
#     * SIMD
#     * Multithreading
#     * Tasks
#     * Multi-process
#         * Shared memory
#         * Distributed memory
#     * GPU programming
#
#
# ## Why so many kinds of parallelism?
#
# * Not all problems are created equal
# * Not all computing machines are created equal
# * We want to maximize comuting while minimizing overhead
#     * Chosen solution will depend upon the amount of computing in each inner loop
#       and the amount of syncronization that is required between loops.
