# # Parallel Computing with Julia
#
# This course will cover:
#
# * Introduction to parallelism
#     * What is happening to our computers?
#
# * Parallelism strategies
#     * SIMD
#     * Multi-threading
#     * Tasks
#     * Multi-process
#         * Shared memory
#         * Distributed memory
#     * GPU programming
#
# * Challenges of parallel computing
#     * Order of execution
#         * execution of out order of Possibility
#         * simultaneous access and mutation
#     * Data access and movement
#     * Code access and movement
#     * Appropriately matching the parallelism strategy to your machine capabilities
#     * Appropriately matching the parallelism strategy with the problem at hand

#-

# ## What is happening to our computers!?
#
# ![](images/40-years-processor-trend.png)
#
# Not only have we gained multiple cores, but processors have become extremely
# complex, with multiple levels of caches, pipelines, predictions, speculations...
#
# ## What is hard about parallel computing
#   * We don't think in parallel
#   * We learn to write and reason about programs serially
#   * The desire for parallelism often comes _after_ you've written your algorithm (and found it too slow!)
#
# ## Summary:
#   * Current computer archetectures push us towards parallel programming for peak performance — even if we're not on a cluster!
#   * But it's hard to design good parallel algorithms
#   * And it's hard to express and reason about those algorithms
