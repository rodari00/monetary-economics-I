module ss_solution

using Plots
using Distributions
using StatsBase
using NLsolve
using LinearAlgebra
using LinearSolve
using Interpolations
using CategoricalArrays

include("C:\\Users\\feder\\Dropbox\\Github\\monetary-economics-I\\modules\\00-utils.jl")
using .utils

export discretize_assets, discretize_productivity,
       backward_iteration, policy_ss, forward_policy, forward_iteration,
       distribution_ss


# Krokecker sum
⊕(x,y) = x .+ y  # we need it to construct the RHS of the FOC to have a n_s x n_a state space matrix where for each y(s)
                # I sum over all possible asset states.

### Asset Discretization ------------------------------------
function discretize_assets(amin, amax, n_a)
    # find maximum ubar of uniform grid corresponding to desired maximum amax of asset grid
    ubar = log(1 + log(1 + amax - amin))
    
    # make uniform grid
    u_grid = linspace(0, ubar, n_a)
    
    # double-exponentiate uniform gid and add amin to get grid from amin to amax
    return amin .+ exp.(exp.(u_grid) .- 1) .- 1
end

### Productivity discretization ----------------------------

function discretize_productivity(ρ, σ, n_s)
    # choose inner-switching probability p to match persistence ρ
    p = (1+ρ)/2
    
    # start with states from 0 to n_s-1, scale by alpha to match standard deviation σ
    lns = range(0,n_s-1,step=1)
    lns *= 2*σ/sqrt(n_s-1)
    
    # obtain Markov transition matrix Pi and its stationary distribution
    Pi = rouwenhorst_Pi(n_s, p)
    pi_star = stationary_markov(Pi,n_s)

    # lns is log , get s and scale so that mean is 1
    s = exp.(lns)
    s /= dot(pi_star, s)
    
    return s, pi_star, Pi

end

### Backward Iteration ----------------------------

function backward_iteration(Va, Pi, a_grid, s, r, β, Y, eis)

    # (i) Take expectation and discount RHS of the first order condition
    Wa = β*Pi*Va

    # (ii) Solve for asset policy
    c_endog = Wa.^(-eis)
    coh = ⊕(reshape(s*Y,:,1),(1+r)*reshape(a_grid,1,:)) # coh = cash on hand (i.e. available income today)

    # Initialize tomorrow's asset grid and loop over each row of the state space matrix
    a = zeros(size(Va)) 
    for s in 1:size(a)[1]
        # Will extrapolate if out of domain bounds
        interp_linear = linear_interpolation(c_endog[s,:] + a_grid,a_grid,extrapolation_bc=Line())
        a[s, :] = interp_linear.(coh[s,:])
    end

    # (iii) Enforce borrowing constraint and back out consumption
    a[a .< a_grid[1]] .= a_grid[1] # need to replace asset when it goes below the borrowing limit (i.e. the borrowing constraint binds)
    c = coh - a

    # (iv) Envelope condition to back up the derivative of the value function
    Va = (1+r) * c.^(-1/eis)

    return (Va, a, c)

end

### Steady State solution --------------------------------
function policy_ss(Pi, a_grid, s, Y, r, β, eis, coh_share = 0.1, tol=1E-9, maxiter = 10000)

    # initial guess for Va: assume consumption x% of cash-on-hand, then get Va from envelope condition
    coh = ⊕(reshape(s*Y,:,1),(1+r)*reshape(a_grid,1,:))
    c = coh_share*coh
    Va = (1+r) * c.^(-1/eis)
    a_old = zeros(size(c)[1],size(c)[2]) .+ 1000.0 # initialize asset space grid with very high value
    iter = 1
    normdiff = 10
    ss_error = zeros(maxiter,1)
    
    a = nothing # BUG!!!?????
    while normdiff > tol && iter <= maxiter
        Va, a, c = backward_iteration(Va, Pi, a_grid, s, r, β, Y, eis)
        normdiff = norm(a - a_old)
        ss_error[iter] = normdiff
        # replace and continue
        a_old = copy(a)
        iter = iter + 1
    
        if iter%100 == 0
            println("Running $iter iteration, with error $normdiff...\n")
        end
    end
    
    # Out of the conditional statement
    if iter < maxiter
        println("Steady State found in $iter iteration!\n")
    else
        println("Reached maximum number of iterations!\n")
    end
    
    return(Va, a, c, normdiff, ss_error)
end



# ---------------------------------------------------------
# Forward iteration

# (1)
function forward_policy(D, a_policy,a_grid)

    # Initialize the matrix
    Dend = zeros(size(D))

    for s in 1:size(a_policy)[1]
        for a in 1:size(a_policy)[2]

            # Impute using Young's method (from utils.jl)
            a_value, a_low, a_high, a_pi = get_lottery(a_policy[s,a],
                                                        a_grid)

            # send pi(s,a) of the mass to gridpoint i(s,a)
            Dend[s, a_low] += a_pi*D[s,a]

            # send 1-pi(s,a) of the mass to gridpoint i(s,a)+1
             Dend[s,a_high] += (1-a_pi)*D[s,a]

        end # end loop over assets
    end # end loop over income

    return Dend

end # end function

# (2)
function forward_iteration(D,Pi,a_policy,a_grid)

    # From D(s,a') to D(s',a') (how many possible ways lead me to the current state)
    Dend = forward_policy(D,a_policy,a_grid)

    return Pi'*Dend
end

# (3)
function distribution_ss(Pi,a_policy,a_grid, tol = 1E-10, maxiter = 10000)

    n_a = length(a_grid)
    n_s = size(Pi)[1]
    # Initialize D and find stationary distribution
    pi_star = stationary_markov(Pi,n_s)
    D = broadcast(*,reshape(pi_star,:,1) ,reshape(ones(n_a)./n_a,1,:))
    
    # Initialize loop
    normdiff = 100
    iter = 1
    while normdiff > tol && iter <= maxiter
        D_new = forward_iteration(D,Pi,a_policy,a_grid)
        normdiff = norm(D_new - D)
        iter = iter + 1
        D = copy(D_new)
    
        if iter%100 == 0
            println("Running $iter iteration, with error $normdiff...\n")
        end
    end

    
    # Out of the conditional statement
    if iter < maxiter
        println("Steady State found in $iter iteration!\n")
    else
        println("Reached maximum number of iterations!\n")
    end
    
    return(D)

end


end