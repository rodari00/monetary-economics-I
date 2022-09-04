
using Plots
using Distributions
using StatsBase
using NLsolve
using LinearAlgebra
using LinearSolve
using Interpolations

include("./utils.jl")
using .utils
include("./ss_solution.jl")
using .ss_solution


### Global Parameters Setup ---------------------------
# remember to always have s on the y axis, and a on the x axis
β = 0.988;
n_s = 7;
n_a = 500;
amax = 200.0;
amin = 0.0;
ρ,σ = 0.975,0.7;
r = 0.01/4;
Y = 1;
eis = 1; # log utility implies EIS = 1

### 1A) Asset Discretization --------------------------


### 1B) Productivity discretization --------------------------

### (2) Backward iteration ----------------------
# Krokecker sum
⊕(x,y) = x .+ y  # we need it to construct the RHS of the FOC to have a n_s x n_a state space matrix where for each y(s)
                # I sum over all possible asset states.

# Generate the grid (double exponential method)
a_grid = discretize_assets(amin, amax, 500);
# Generate productivity grid
s, pi_star, Pi = discretize_productivity(ρ, σ, n_s)


Va, a, c,normdiff,ss_error= policy_ss(Pi, a_grid, y, r, β, eis)



# Plot Policy function c(a)
ytmp = round(s[1], digits = 2)
policy_fun = plot(a_grid,c[1,:],label = "y = $ytmp")
for state in 2:n_s
    ytmp = round(s[state], digits = 2)
    plot!(a_grid, c[state,:], label = "y = $ytmp")
end
policy_fun


# Plot Savings function (a'-a)(a)
ytmp = round(s[1], digits = 2)
savings_fun = plot(a_grid,a[1,:]-a_grid,label = "y = $ytmp")
for state in 2:n_s
    ytmp = round(s[state], digits = 2)
    plot!(a_grid, a[state,:]-a_grid, label = "y = $ytmp")
end
savings_fun