# Neural Ordinary Differential Equations with Flux

All the tools of SciMLSensitivity.jl can be used with Lux.jl.

## Using Lux `Chain` neural networks with Optimization.jl

Let's use this to build and train a neural ODE from scratch. In this example, we will
optimize both the neural network parameters `p` and the input initial condition `u0`.
Notice that Optimization.jl works on a vector input, so we have to concatenate `u0`
and `p` and then in the loss function split to the pieces.

If you have a Flux layer, you can use
[Lux.transform](https://lux.csail.mit.edu/dev/api/Lux/flux_to_lux#Lux.transform) to convert
it to a lux layer. This is automatically handled if you are using DiffEqFlux.jl

```@example neuralode2
using Lux, OrdinaryDiffEq, SciMLSensitivity, Optimization, OptimizationOptimisers,
    OptimizationNLopt, Plots, Zygote, Random, ComponentArrays

u0 = [2.0; 0.0]
datasize = 30
tspan = (0.0, 1.5)

function trueODEfunc(du, u, p, t)
    true_A = [-0.1 2.0; -2.0 -0.1]
    du .= ((u .^ 3)'true_A)'
end
t = range(tspan[1], tspan[2], length = datasize)
prob = ODEProblem(trueODEfunc, u0, tspan)
ode_data = Array(solve(prob, Tsit5(), saveat = t))

dudt2 = Chain(x -> x .^ 3, Dense(2, 50, tanh), Dense(50, 2))
ps, st = Lux.setup(Random.default_rng(), dudt2)
ps = ComponentArray(ps)
dudt2 = Lux.Experimental.StatefulLuxLayer(dudt2, ps, st)

dudt(u, p, t) = dudt2(u, p)
prob = ODEProblem(dudt, u0, tspan, ps)

θ = ComponentArray(; u0, ps)

function predict_n_ode(θ)
    Array(solve(prob, Tsit5(), u0 = θ.u0, p = θ.ps, saveat = t))
end

function loss_n_ode(θ)
    pred = predict_n_ode(θ)
    loss = sum(abs2, ode_data .- pred)
    loss, pred
end

loss_n_ode(θ)

callback = function (θ, l, pred; doplot = false) #callback function to observe training
    display(l)
    # plot current prediction against data
    pl = scatter(t, ode_data[1, :], label = "data")
    scatter!(pl, t, pred[1, :], label = "prediction")
    display(plot(pl))
    return false
end

# Display the ODE with the initial parameter values.
callback(θ, loss_n_ode(θ)...)

# use Optimization.jl to solve the problem
adtype = Optimization.AutoZygote()

optf = Optimization.OptimizationFunction((p, _) -> loss_n_ode(p), adtype)
optprob = Optimization.OptimizationProblem(optf, θ)

result_neuralode = Optimization.solve(optprob, Adam(0.05); callback = callback,
    maxiters = 300)
```

Notice that the advantage of this format is that we can use Optim's optimizers, like
`LBFGS` with a full `Chain` object, for all of Lux's neural networks, like
convolutional neural networks.

![](https://user-images.githubusercontent.com/1814174/51399500-1f4dd080-1b14-11e9-8c9d-144f93b6eac2.gif)
