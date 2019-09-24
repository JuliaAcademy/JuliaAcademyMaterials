import Pkg; Pkg.add(Pkg.PackageSpec(url="https://github.com/JuliaComputing/JuliaAcademyData.jl"))
using JuliaAcademyData; activate("Parallel_Computing")

# # A brief introduction to Tasks
#
# You're working on a computer that's doing _lots_ of things. It's managing
# inputs, outputs, delegating control of the CPU between Julia and _all_ of
# the other applications you have running. This wasn't always the case — does
# anyone remember the days before you could just switch between applications?
#
# It's not really doing all these things at once, but for the most part it
# gives the _appearance_ of parallelism. We think about our computers as doing
# _lots_ of things simultaneously — but it's not really simultaneous. It's just
# switching between tasks so fast that it feels simultaneous.
#
# This kind of task switching is perfect for situations like an operating system
# where you're just waiting for user input most of the time. The OS multitasking
# you're familiar with is called "preemptive" multitasking — the operating system
# sits at the top and can arbitrarily control who gets to run when. Julia's task
# system uses cooperative multitasking (also known as coroutines or green threads).

#-

# Tasks work best when they're waiting for some _external_ condition to complete
# their work. Let's say we had a directory "results" and wanted to process any
# new files that appeared there:

using FileWatching
isdir("results") || mkdir("results")
watch_folder("results", #= time out in seconds =# 5)

# Julia happily will sit there and wait for something to happen... but it's
# blocking anything else from happening while it's doing so! This is the perfect
# case for a Task. We can say we want a given expression to run asynchronously
# in a Task with the `@async` macro

t = @async watch_folder("results") # no timeout means it will wait forever!

#-

run(`touch results/0.txt`)

#-

file, info = fetch(t)
file # |> process

# We can even bundle this up into a repeating task:

isdone = false
function process_folder(dir)
    !isdir("processed-results") && mkdir("processed-results")
    while !isdone
        file, info = watch_folder(dir)
        path = joinpath(dir, file)
        if isfile(path)
            print("processing $path...")
            run(`cp $path processed-results/$file`) # Or actually do real work...
        end
    end
end

t = @async process_folder("results")

#-

run(`touch results/1.txt`)
sleep(.1)
readdir("processed-results")

#-

run(`touch results/2.txt`)
sleep(.1)
readdir("processed-results")

#-

isdone = true
run(`touch results/3.txt`)
sleep(.1)
readdir("processed-results")

#-

run(`touch results/4.txt`)
sleep(.1)
readdir("processed-results")

#-

rm("results", recursive=true)
rm("processed-results", recursive=true)

# ## Quiz:
#
# How long will this take?

@time for i in 1:10
    sleep(1)
end

# What about this?

@time for i in 1:10
    @async sleep(1)
end

# And finally, this?

@time @sync for i in 1:10
    @async sleep(1)
end

# Now what if I had something that actually did work?

function work(N)
    series = 1.0
    for i in 1:N
        series += (isodd(i) ? -1 : 1) / (i*2+1)
    end
    return 4*series
end
work(1)
@time work(100_000_000)

#-

@time @sync for i in 1:10
    @async work(100_000_000)
end

# # So what's happening here?
#
# `sleep` is nicely cooperating with our tasks

methods(sleep)

# # Fetching values from tasks

#-

# You can even fetch values from tasks

t = @async (sleep(5); rand())

#-

wait(t)

#-

fetch(t)

# # Key takeaways
#
# There is a lot more to tasks, but they form the foundation for reasoning about
# actually _doing_ computation in parallel (and not just hoping that things will
# cooperate for us to emulate parallelism by task switching).
#
# * `@async` creates and starts running a task
# * `@sync` waits for them to all complete
# * We can reason about something that runs asynchronously and may return a value
#   at some point in the future with `fetch`. Or we can just `wait` for it.

