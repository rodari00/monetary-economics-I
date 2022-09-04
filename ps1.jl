
using Plots
using Distributions
using StatsBase
using NLsolve
using LinearAlgebra
using LinearSolve
using Interpolations

### Global Parameters Setup ---------------------------
# remember to always have s on the y axis, and a on the x axis
β = 0.988
n_s = 7
n_a = 500
amax = 200.0
amin = 0.0
ρ,σ = 0.975,0.7
r = 0.01/4
Y = 1
eis = 1 # log utility implies EIS = 1

### 1A) Asset Discretization --------------------------

# Matlab like function
function linspace(z_start::Real, z_end::Real, z_n::Int64)
    return collect(range(z_start,stop=z_end,length=z_n))
end


function discretize_assets(amin, amax, n_a)
    # find maximum ubar of uniform grid corresponding to desired maximum amax of asset grid
    ubar = log(1 + log(1 + amax - amin))
    
    # make uniform grid
    u_grid = linspace(0, ubar, n_a)
    
    # double-exponentiate uniform gid and add amin to get grid from amin to amax
    return amin .+ exp.(exp.(u_grid) .- 1) .- 1
end

### 1B) Productivity discretization --------------------------

# Rouwenhorst method
function rouwenhorst_Pi(N, p)
    # base case Pi_2
    Pi =[p (1-p);
        p (1-p)]
    
    # recursion to build up from Pi_2 to Pi_N
    for n in 3:N
        Pi_old = Pi
        Pi = zeros(n, n)
        
        Pi[1:end-1, 1:end-1] += p * Pi_old
        Pi[1:end-1, 2:end] += (1 - p) * Pi_old
        Pi[2:end, 1:end-1] += (1 - p) * Pi_old
        Pi[2:end, 2:end] += p * Pi_old
        Pi[2:end-1, :] /= 2
    end

    return Pi
    
end


# Find stationary distribution
function stationary_markov(Pi, tol=1E-14, maxiter = 10000)
    # start with uniform distribution over all states
    n = size(Pi)[1]
    pi_old = fill(1/n, n, 1);
    normdiff = Inf
    iter = 1
    
    # update distribution using Pi until successive iterations differ by less than tol
    while normdiff > tol && iter <= maxiter
        pi_new = Pi'*pi_old # the f(v) map
        normdiff = norm(pi_new - pi_old)
        # replace and continue
        pi_old = pi_new
        iter = iter + 1
    end
    return (pi_old, normdiff, iter) # returns a tuple
end


function discretize_productivity(ρ, σ, n_s)
    # choose inner-switching probability p to match persistence ρ
    p = (1+ρ)/2
    
    # start with states from 0 to n_s-1, scale by alpha to match standard deviation σ
    lns = range(0,n_s-1,step=1)
    lns *= 2*σ/sqrt(n_s-1)
    
    # obtain Markov transition matrix Pi and its stationary distribution
    Pi = rouwenhorst_Pi(n_s, p)
    pi_star,error,iter = stationary_markov(Pi)

    # lns is log , get s and scale so that mean is 1
    s = exp.(lns)
    s /= dot(pi_star, s)
    
    return s, pi_star, Pi
end

### (2) Backward iteration ----------------------
# Krokecker sum
⊕(x,y) = x .+ y  # we need it to construct the RHS of the FOC to have a n_s x n_a state space matrix where for each y(s)
                # I sum over all possible asset states.


function backward_iteration(Va, Pi, a_grid, s, r, β, eis)

    # (i) Take expectation and discount RHS of the first order condition
    Wa = β*Pi*Va

    # (ii) Solve for asset policy
    c_endog = Wa.^(-eis)
    coh = ⊕(reshape(s*Y,:,1),(1+r)*reshape(a_grid,1,:)) # coh = cash on hand (i.e. available income today)

    # Initialize tomorrow's asset grid and loop over each row of the state space matrix
    a = zeros(n_s,n_a) 
    for s in 1:n_s
        # Will extrapolate if out of domain bounds
        interp_linear = linear_interpolation(coh[s,:],c_endog[s,:],extrapolation_bc=Line())
        a[s, :] = interp_linear.(reshape(a_grid,1,:))
    end

    # (iii) Enforce borrowing constraint and back out consumption
    a[a .< a_grid[1]] .= a_grid[1] # need to replace asset when it goes below the borrowing limit (i.e. the borrowing constraint binds)
    c = coh - a

    # (iv) Envelope condition to back up the derivative of the value function
    Va = (1+r) * c.^(-1/eis)

    return Va, a, c

end



function policy_ss(Pi, a_grid, y, r, β, eis, coh_share = 0.1, tol=1E-9, maxiter = 10000)

    # initial guess for Va: assume consumption x% of cash-on-hand, then get Va from envelope condition
    coh = ⊕(reshape(s*Y,:,1),(1+r)*reshape(a_grid,1,:))
    c = coh_share*coh
    Va = (1+r) * c.^(-1/eis)
    a_old = zeros(size(c)[1],size(c)[2])  # initialize asset space grid with very high value
    iter = 1
    normdiff = 10
    ss_error = zeros(maxiter,1)
    
    while normdiff > tol && iter <= maxiter
        Va, a, c = backward_iteration(Va, Pi, a_grid, s, r, β, eis)
        normdiff = norm(a - a_old)
        ss_error[iter] = normdiff
        # replace and continue
        a_old = a
        iter = iter + 1

        if iter%500 == 0
            println("Running $iter iteration, with error $normdiff...\n")
        end
    end
    
    # Out of the conditional statement
    if iter < maxiter
        println("Steady State found in $iter iteration!")
    else
        println("Reached maximum number of iterations!")
    end

    return(Va, a, c,normdiff, ss_error)
end



# Generate the grid (double exponential method)
a_grid = discretize_assets(amin, amax, 500);
# Generate productivity grid
s, pi_star, Pi = discretize_productivity(ρ, σ, n_s)


Va, a, c,normdiff,ss_error = policy_ss(Pi, a_grid, y, r, β, eis)







for s in 1:7
    if s == 1
     plot(a_grid[1:120],c[s,1:120])
    else
    plot!(a_grid[1:120],c[s,1:120])
    end
end
    