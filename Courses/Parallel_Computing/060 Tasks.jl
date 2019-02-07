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

# Tasks work best when they're waiting for some _external_ condition to complete
# their work. Let's say we had a directory "results" and wanted to process any
# new files that appeared there:

using FileWatching
mkdir("results")
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
            run(`cp $path processed-results/$file`) # Or actually do real work...
        end
    end
end

t = @async process_folder("results")

#-
readdir("results")
#-
run(`touch results/2.txt`)
readdir("processed-results")
#-
isdone = true
run(`touch results/3.txt`)
readdir("processed-results")
#-
run(`touch results/4.txt`)
readdir("processed-results")
#-
t
#-
# ## Quiz:
#
# How long will this take?

@time for i in 1:10
    sleep(1)
end
