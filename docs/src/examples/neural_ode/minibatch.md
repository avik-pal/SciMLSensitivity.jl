# Training a Neural Ordinary Differential Equation with Mini-Batching

```@example
using OrdinaryDiffEq, Lux, SciMLSensitivity, OrdinaryDiffEq, Optimization,
    OptimizationOptimisers, Random, Plots, ComponentArrays, Zygote, IterTools

rng = Random.default_rng()

function newtons_cooling(du, u, p, t)
    temp = u[1]
    k, temp_m = p
    du[1] = dT = -k * (temp - temp_m)
end

function true_sol(du, u, p, t)
    true_p = [log(2) / 8.0, 100.0]
    newtons_cooling(du, u, true_p, t)
end

ann = Chain(Dense(1, 8, tanh), Dense(8, 1, tanh))
θ, st = Lux.setup(rng, ann)
θ = ComponentArray(θ)
ann = Lux.Experimental.StatefulLuxLayer(ann, θ, st)

function dudt_(u, p, t)
    ann(u, p) .* u
end

function predict_adjoint(p, time_batch)
    _prob = remake(prob, u0 = u0, p = p)
    Array(solve(_prob, Tsit5(), saveat = time_batch))
end

function loss_adjoint(p, batch, time_batch)
    pred = predict_adjoint(p, time_batch)
    sum(abs2, reduce(hcat, batch) - pred), pred
end

u0 = Float32[200.0]
datasize = 30
tspan = (0.0f0, 3.0f0)

t = range(tspan[1], tspan[2], length = datasize)
true_prob = ODEProblem(true_sol, u0, tspan)
ode_data = Array(solve(true_prob, Tsit5(), saveat = t))

prob = ODEProblem{false}(dudt_, u0, tspan, θ)

k = 10
train_loader = zip(Iterators.partition(eachcol(ode_data), k), Iterators.partition(t, k))

for (x, y) in train_loader
    @show reduce(hcat, x)
    @show y
end

numEpochs = 300
losses = []
function cb(p, l, pred)
    push!(losses, l)
    @show l
    pl = scatter(t, ode_data[1, :], label = "data", color = :black, ylim = (150, 200))
    scatter!(pl, t, pred[1, :], label = "prediction", color = :darkgreen)
    display(plot(pl))
    false
end

opt_prob = OptimizationProblem(OptimizationFunction((θ, p, data_batch, time_batch) -> loss_adjoint(θ, data_batch, time_batch), AutoZygote()), θ)
res = solve(opt_prob, Adam(0.001), IterTools.ncycle(train_loader, 300), callback = cb)

# Now lets see how well it generalizes to new initial conditions 

starting_temp = collect(10:30:250)
true_prob_func(u0) = ODEProblem(true_sol, [u0], tspan)
color_cycle = palette(:tab10)
pl = plot()
for (j, temp) in enumerate(starting_temp)
    ode_test_sol = solve(ODEProblem(true_sol, [temp], (0.0f0, 10.0f0)), Tsit5(),
        saveat = 0.0:0.5:10.0)
    ode_nn_sol = solve(ODEProblem{false}(dudt_, [temp], (0.0f0, 10.0f0), res.u), Tsit5())
    scatter!(pl, ode_test_sol, var = (0, 1), label = "", color = color_cycle[j])
    plot!(pl, ode_nn_sol, var = (0, 1), label = "", color = color_cycle[j], lw = 2.0)
end
display(pl)
title!("Neural ODE for Newton's Law of Cooling: Test Data")
xlabel!("Time")
ylabel!("Temp")
```

When training a neural network, we need to find the gradient with respect to our data set. There are three main ways to partition our data when using a training algorithm like gradient descent: stochastic, batching and mini-batching. Stochastic gradient descent trains on a single random data point each epoch. This allows for the neural network to better converge to the global minimum even on noisy data, but is computationally inefficient. Batch gradient descent trains on the whole data set each epoch and while computationally efficient is prone to converging to local minima. Mini-batching combines both of these advantages and by training on a small random "mini-batch" of the data each epoch can converge to the global minimum while remaining more computationally efficient than stochastic descent. Typically, we do this by randomly selecting subsets of the data each epoch and use this subset to train on. We can also pre-batch the data by creating an iterator holding these randomly selected batches before beginning to train. The proper size for the batch can be determined experimentally. Let us see how to do this with Julia.

For this example, we will use a very simple ordinary differential equation, newtons law of cooling. We can represent this in Julia like so.

```@example minibatch
using OrdinaryDiffEq, Lux, SciMLSensitivity, OrdinaryDiffEq, Optimization,
    OptimizationOptimisers, Random, Plots, ComponentArrays, Zygote, IterTools

rng = Random.default_rng()
function newtons_cooling(du, u, p, t)
    temp = u[1]
    k, temp_m = p
    du[1] = dT = -k * (temp - temp_m)
end

function true_sol(du, u, p, t)
    true_p = [log(2) / 8.0, 100.0]
    newtons_cooling(du, u, true_p, t)
end
```

Now we define a neural-network using a linear approximation with 1 hidden layer of 8 neurons.

```@example minibatch

ann = Chain(Dense(1, 8, tanh), Dense(8, 1, tanh))
θ, st = Lux.setup(rng, ann)
θ = ComponentArray(θ)
ann = Lux.Experimental.StatefulLuxLayer(ann, θ, st)

function dudt_(u, p, t)
    ann(u, p) .* u
end
```

From here we build a loss function around it.

```@example minibatch
function predict_adjoint(p, time_batch)
    _prob = remake(prob, u0 = u0, p = p)
    Array(solve(_prob, Tsit5(), saveat = time_batch))
end

function loss_adjoint(p, batch, time_batch)
    pred = predict_adjoint(p, time_batch)
    sum(abs2, reduce(hcat, batch) - pred), pred
end
```

To add support for batches of size `k`. To use this we pass in the `ode_data` and `t` as the 'x' and 'y' data to batch respectively. The parameter `batchsize` controls the size of our batches. We check our implementation by iterating over the batched data.

```@example minibatch
u0 = Float32[200.0]
datasize = 30
tspan = (0.0f0, 3.0f0)

t = range(tspan[1], tspan[2], length = datasize)
true_prob = ODEProblem(true_sol, u0, tspan)
ode_data = Array(solve(true_prob, Tsit5(), saveat = t))

prob = ODEProblem{false}(dudt_, u0, tspan, θ)

k = 10
train_loader = zip(Iterators.partition(eachcol(ode_data), k), Iterators.partition(t, k))

for (x, y) in train_loader
    @show reduce(hcat, x)
    @show y
end

numEpochs = 300
losses = []
function cb(p, l, pred)
    push!(losses, l)
    @show l
    pl = scatter(t, ode_data[1, :], label = "data", color = :black, ylim = (150, 200))
    scatter!(pl, t, pred[1, :], label = "prediction", color = :darkgreen)
    display(plot(pl))
    false
end

opt_prob = OptimizationProblem(OptimizationFunction((θ, p, data_batch, time_batch) -> loss_adjoint(θ, data_batch, time_batch), AutoZygote()), θ)
res = solve(opt_prob, Adam(0.001), IterTools.ncycle(train_loader, 300), callback = cb)
```

Finally, we can see how well our trained network will generalize to new initial conditions.

```@example minibatch
starting_temp = collect(10:30:250)
true_prob_func(u0) = ODEProblem(true_sol, [u0], tspan)
color_cycle = palette(:tab10)
pl = plot()
for (j, temp) in enumerate(starting_temp)
    ode_test_sol = solve(ODEProblem(true_sol, [temp], (0.0f0, 10.0f0)), Tsit5(),
        saveat = 0.0:0.5:10.0)
    ode_nn_sol = solve(ODEProblem{false}(dudt_, [temp], (0.0f0, 10.0f0), res.u), Tsit5())
    scatter!(pl, ode_test_sol, var = (0, 1), label = "", color = color_cycle[j])
    plot!(pl, ode_nn_sol, var = (0, 1), label = "", color = color_cycle[j], lw = 2.0)
end
display(pl)
title!("Neural ODE for Newton's Law of Cooling: Test Data")
xlabel!("Time")
ylabel!("Temp")
```
