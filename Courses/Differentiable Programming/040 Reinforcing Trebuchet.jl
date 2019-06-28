# # How To Aim Your Flagon

#-

# ## Loading your Trebuchet

#-

# Today we practice the ancient medieval art of throwing stuff. First up, we load our trebuchet simulator, Trebuchet.jl.

using Trebuchet

# We can see what the trebuchet looks like, by explicitly creating a trebuchet state, running a simulation, and visualising the trajectory.

t = TrebuchetState()
simulate(t)
visualise(t)

# For training and optimisation, we don't need the whole visualisation, just a simple function that accepts and produces numbers. The `shoot` function just takes a wind speed, angle of release and counterweight mass, and tells us how far the projectile got.

function shoot(wind, angle, weight)
    Trebuchet.shoot((wind, Trebuchet.deg2rad(angle), weight))[2]
end

#-

shoot(0, 30, 400)

# It's worth playing with these parameters to see the impact they have. How far can you throw the projectile, tweaking only the angle of release?

#-

# There's actually a much better way of aiming the trebuchet. Let's load up a machine learning library, Flux, and see what we can do.

pathof(Trebuchet)

#-

using Flux, Trebuchet
using Flux.Tracker: gradient, forwarddiff

# Firstly, we're going to wrap `shoot` to take a _parameter vector_ (just a list of the three numbers we're interested in). There's also a call to `forwarddiff` here, which tells Flux to differentiate the trebuchet itself using forward mode. The number of parameters is small, so forward mode will be the most efficient way to do it. Otherwise Flux defaults to reverse mode.

shoot(ps) = forwarddiff(p -> shoot(p...), ps)

# We can get a distance as usual.

shoot([0, 45, 200])

# But we can also get something much more interesting: *gradients* for each of those parameters with respect to distance.

gradient(shoot, [0, 45, 200])

# What does these numbers mean? The gradient tells us, very roughly, that if we increase a parameter – let's say we make wind speed 1 m/s stronger – distance will also increase by about 4 metres. Let's try that.

shoot([1, 45, 200])

# Lo and behold, this is indeed about four metres further!

shoot([1, 45, 200]) - shoot([0, 45, 200])

# So this seems like very useful information if we're trying to aim, or maximise distance. Notice that our gradient for the release angle is negative – increasing angle will decrease distance, so in other words we should probably *decrease* angle if we want more distance. Let's try that.

shoot([0, 10, 200])

# Oh no, this is actually *less* far than before!

#-

# So if the angle is too shallow, the projectile doesn't spend enough time in the air to gain any distance before hitting the ground. But if it's too high, the projectile doesn't have enough horizontal speed even with lots of time in the air. So we'll have to find a middle ground.
#
# More generally, the lesson here is that the gradient only gives you limited information; it helps us take a small step towards a better aim, and we can keep iterating to get to the best possible aim. For example, we choose a starting angle:

angle = 45
shoot([0, angle, 200])

# Get a gradient for `angle` alone:

dangle = gradient(angle -> shoot(Tracker.collect([0, angle, 200])), angle)[1] |> Flux.data

# Update the angle, using the learning rate η:

η = 10
angle += η*dangle

#-

shoot([0, angle, 200])

# Now we just lather, rinse and repeat! Ok, maybe we should write a loop to automate this a bit.

for i = 1:10
    dangle = gradient(angle -> shoot(Tracker.collect([0, angle, 200])), angle)[1] |> Flux.data
    angle += η*dangle
    @show angle
end
shoot([0, angle, 200])

# Notice how the change in the angle slows down as things converge. Turns out the best angle is about 30 degrees, and we can hit about 90 metres.
#
# We can make this nicely repeatable and get the best angle for any given wind speed.

function best_angle(wind)
    angle = 45
    objective(angle) = shoot(Tracker.collect([wind, angle, 200]))
    for i = 1:10
        dangle = gradient(objective, angle)[1] |> Flux.data
        angle += η*dangle
    end
    return angle
end

#-

best_angle(0)

#-

best_angle(10)

#-

best_angle(-10)

# It turns out that if the wind is on our side, we should just throw the projectile upwards and let it get blown along. If the wind is strong against us, just chuck that stone right into it.

t = TrebuchetState(release_angle = deg2rad(19), wind_speed = -10)
simulate(t)
visualise(t)

# ## Accuracy Matters

#-

# In optimisation terms, we just created an objective (distance) and tried to maximise that objective. Flinging boulders as far as possible has its moments, but lacks a certain subtlety. What if we instead want to hit a precise target?

t = TrebuchetState()
simulate(t)
visualise(t, 50)

# The way to do this is to state the problem in terms of maximising, or minisming, some number – the objective. In this case, an easy way to come up with an objective is to take the difference from our target (gets closer to 0 as aim gets better) and square it (so it's always positive: 0 is the lowest *and* best possible score).

#-

# Here's a modified `best_angle` function that takes a target and tells us the distance it acheived.

η = 0.1
function best_angle(wind, target)
    angle = 45
    objective(angle) = (shoot(Tracker.collect([wind, angle, 200])) - target)^2
    for i = 1:30
        dangle = gradient(objective, angle)[1] |> Flux.data
        angle -= η*dangle
    end
    return angle, shoot([wind, angle, 200])
end

# It's pretty accurate!

best_angle(0, 50)

# Even when we try to push it, by making wind really strong.

best_angle(-20, 35)

#-

t = TrebuchetState(release_angle = deg2rad(21.8), weight = 200, wind_speed = -20)
simulate(t)
visualise(t, 35)

# ## Siege Weapon Autopilot

#-

# Finally, we go one level more meta by training a neural network to aim the trebuchet for us. Rather than solving a whole optimisation problem every time we want to aim, we can just ask the network for good parameters and get them in constant time.
#
# Here's a simple multi layer perceptron. Its input is two parameters (wind speed and target) and its output is two more (release angle and counterweight mass).

model = Chain(Dense(2, 16, σ),
              Dense(16, 64, σ),
              Dense(64, 16, σ),
              Dense(16, 2)) |> f64

θ = params(model)

function aim(wind, target)
    angle, weight = model([wind, target])
    angle = σ(angle)*90
    weight = weight + 200
    angle, weight
end

distance(wind, target) = shoot(Tracker.collect([wind, aim(wind, target)...]))

# The model's initial guesses will be fairly random, and miss the mark.

aim(0, 70)

#-

distance(0, 70)

# However, just as before, we can define an objective – or loss – and get gradients.

function loss(wind, target)
    try
        (distance(wind, target) - target)^2
    catch e
        # Roots.jl sometimes give convergence errors, ignore them
        param(0)
    end
end

loss(0, 70)

# This time, though, we'll get gradients for the *model parameters*, and updating these will improve the network's accuracy. This works because we're able to differentiate the *whole program*; the backwards pass propagates errors through the trebuchet simulator and then through the ML model.

dθ = gradient(θ) do
    loss(0, 70)
end
dθ[model[1].W]

#-

DIST  = (20, 100) # Maximum target distance
SPEED = 5         # Maximum wind speed

lerp(x, lo, hi) = x*(hi-lo)+lo

randtarget() = (randn() * SPEED, lerp(rand(), DIST...))

#-

using Statistics

meanloss() = mean(sqrt(loss(randtarget()...)) for i = 1:100)

opt = ADAM()

dataset = (randtarget() for i = 1:10_000)

Flux.train!(loss, θ, dataset, opt, cb = Flux.throttle(() -> @show(meanloss()), 10))

# After only a few minutes of training, we're getting solid accuracy, even on hard wind speeds and targets. You can run the training loop again to improve the accuracy even further.

wind, target = -10, 50
angle, mass = Flux.data.(aim(wind, target))
t = TrebuchetState(release_angle = deg2rad(angle), weight = mass, wind_speed = wind)
simulate(t)
visualise(t, target)

# Notice that aiming with a neural net in one shot is significantly faster than solving the optimisation problem; and we only have a small loss in accuracy.

@time aim(wind, target)

#-

@time best_angle(wind, target)

#-
