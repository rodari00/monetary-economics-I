
using Plots
using Distributions
using StatsBase
using NLsolve
using LinearAlgebra
using LinearSolve
using Interpolations
using CategoricalArrays

#=
include("utils.jl")
using .utils
include("01-ss_solution.jl")
using .ss_solution
=#

## Global Parameters Setup ---------------------------
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


### 0) Utils ----------------------------------------------

# ---------------------------------------------------------
# Linear spacing
function linspace(z_start::Real, z_end::Real, z_n::Int64)
    return collect(range(z_start,stop=z_end,length=z_n))
end

# ---------------------------------------------------------
# Rouwenhorst method
function rouwenhorst_Pi(N, p)
    # base case Pi_2
    Pi =[p (1-p);
        (1-p) p]
    
    # recursion to build up from Pi_2 to Pi_N
    for n in 3:N
        Pi_old = copy(Pi)
        Pi = zeros(n, n)
        
        Pi[1:end-1, 1:end-1] += p * Pi_old
        Pi[1:end-1, 2:end] += (1 - p) * Pi_old
        Pi[2:end, 1:end-1] += (1 - p) * Pi_old
        Pi[2:end, 2:end] += p * Pi_old
        Pi[2:end-1, :] /= 2
    end

    return Pi
    
end

# ---------------------------------------------------------
# Markov stationary distribution
function stationary_markov(Pi)

    # QR decomposition method (shorturl.at/EHTY7)
    A =  [(Matrix(1.0I, n_s, n_s)- Pi)';
        ones(1,n_s)]
    b = [zeros(n_s,1); 1]

    Q,R = qr(A)
    b_prime = Q'*b 
    prob = LinearProblem(R, b_prime)
    pi_star = solve(prob)

    return (pi_star.u) 
end

### 1A) Asset Discretization ------------------------------------
function discretize_assets(amin, amax, n_a)
    # find maximum ubar of uniform grid corresponding to desired maximum amax of asset grid
    ubar = log(1 + log(1 + amax - amin))
    
    # make uniform grid
    u_grid = linspace(0, ubar, n_a)
    
    # double-exponentiate uniform gid and add amin to get grid from amin to amax
    return amin .+ exp.(exp.(u_grid) .- 1) .- 1
end

### 1B) Productivity discretization ----------------------------

function discretize_productivity(ρ, σ, n_s)
    # choose inner-switching probability p to match persistence ρ
    p = (1+ρ)/2
    
    # start with states from 0 to n_s-1, scale by alpha to match standard deviation σ
    lns = range(0,n_s-1,step=1)
    lns *= 2*σ/sqrt(n_s-1)
    
    # obtain Markov transition matrix Pi and its stationary distribution
    Pi = rouwenhorst_Pi(n_s, p)
    pi_star = stationary_markov(Pi)

    # lns is log , get s and scale so that mean is 1
    s = exp.(lns)
    s /= dot(pi_star, s)
    
    return s, pi_star, Pi

end

### 2A) Backward iteration ------------------------------------

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

### 2B) Steady State solution --------------------------------
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
        Va, a, c = backward_iteration(Va, Pi, a_grid, s, r, β, eis)
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

### 2C) Compute the Steady State ------------------------------
# Generate the grid (double exponential method)
a_grid = discretize_assets(amin, amax, 500);
# Generate productivity grid
s, pi_star, Pi = discretize_productivity(ρ, σ, n_s);


Va, a, c, normdiff,ss_error= policy_ss(Pi, a_grid, s, Y, r, β, eis)



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










### 3) Forward iteration

### 3A) Young's method (https://julienpascal.github.io/post/young_2010/)

# Not my function. Credit to: https://discourse.julialang.org/t/findnearest-function/4143/4
function closest_index(a::Vector,x::Real)

    if isempty(a) == true
      error("xGrid is empty in function closest_index.")
    end

    if isnan(x) == true
      error("val is NaN in function closest_index.")
    end

   idx = searchsortedfirst(a,x)
   if (idx==1); return idx; end
   if (idx>length(a)); return length(a); end
   if (a[idx]==x); return idx; end
   if (abs(a[idx]-x) < abs(a[idx-1]-x))
      return idx
   else
      return idx-1
   end
end

# Returns best index and best value
function closest_value_and_index(xGrid::Vector, val::Real)

    # get index
    ibest = closest_index(xGrid, val)

    # Return best value on grid, and the corresponding index
    return xGrid[ibest], ibest

end


function get_lottery(a::Real, a_grid::Vector)

    if isempty(a_grid) == true
        error("The Grid is empty in function get_lottery.")
      end
  
      if isnan(a) == true
        error("a is NaN in function get_lottery.")
      end

    a_min = minimum(a_grid) #Upper bound for the grid
    a_max = maximum(a_grid) #Lower bound for the grid
    n_a = length(a_grid) #Number of points on the grid

    # Project true value on the grid:
    (aValue_proj,aIndex_proj) = closest_value_and_index(a_grid, a)

    # To store the location of the value below and above the true value:
    aIndex_below = 0
    aIndex_above = 0

    # If the true value is above the projection
    if a >= aValue_proj
        aIndex_below = aIndex_proj
        aIndex_above = aIndex_proj + 1
    # If the true value is below the projection
    elseif a < aValue_proj
        aIndex_below = aIndex_proj -1
        aIndex_above = aIndex_proj
    end

    # Boundary cases
    if aIndex_proj == 1
        aIndex_below = 1
        aIndex_above = 2
    elseif aIndex_proj == n_a
        aIndex_below = n_a - 1
        aIndex_above = n_a
    end

    # Special case 1: a < a_min
    if a <= a_min
        a_pi = 1
    elseif a >= a_max
    # Special case 2: a > a_max
        a_pi = 0
    else
        a_pi = 1.0 - ((a - a_grid[aIndex_below])/(a_grid[aIndex_above] - a_grid[aIndex_below]))
        a_pi = min(1.0, max(0.0, a_pi))
    end

    return(aValue_proj, aIndex_below, aIndex_above, a_pi)
end # end of function

function forward_policy(D, a_policy,a_grid)

    # Initialize the matrix
    Dend = zeros(size(D))

    for s in 1:size(a_policy)[1]
        for a in 1:size(a_policy)[2]

            # Impute using Young's method
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


function forward_iteration(D,Pi,a_policy,a_grid)

    # From D(s,a') to D(s',a') (how many possible ways lead me to the current state)
    Dend = forward_policy(D,a_policy,a_grid)

    return Pi'*Dend
end



function distribution_ss(Pi,a_policy,a_grid, tol = 1E-10, maxiter = 10000)

    n_a = length(a_grid)

    # Initialize D and find stationary distribution
    pi_star = stationary_markov(Pi)
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

D = distribution_ss(Pi,a, a_grid)


# To have a proper distribution the sum of D(s,a) must be one,
# so rescale it
F_d = cumsum(sum(D, dims = 1), dims = 2)

# Plot the marginal CDF
plot(a_grid,reshape(F_d,:,1), label = "F(a)")



### Average Asset

A_avg = sum(broadcast(*,D,reshape(a_grid,1,:)))

### Average MPC ------------------------

function mpc(c,a_grid)

    # Initialize matrix
    mpcs = zeros(size(c))

    # Interior forward difference
    mpcs[:,2:end-1] = (1+r)*(c[:,3:end]- c[:,2:end-1])./reshape((a_grid[3:end] -a_grid[2:end-1]),1,:)

    # Boundary differences
    mpcs[:,1] = (1+r)*(c[:,2]- c[:,1])./(a_grid[2] -a_grid[1])
    mpcs[:,end] = (1+r)*(c[:,end]- c[:,end-1])./(a_grid[end] -a_grid[end-1])

    # Enforce unitary MPC at the boundary
    mpcs[a.==a_grid[1]] .= 1

    # Average MPC
    avgmpc = sum(broadcast(*,D,mpcs))

    return(mpcs, avgmpc)
end


 # Asset distribution
 plot(a_grid,D[1,:])

asset_freq = [reshape(a_grid,:,1) reshape(sum(D,dims=1),:,1)]
nbins = 30


function create_histogram(freq_table,nbins)

    bins = cut(freq_table[:,1],nbins)
    grid = unique(bins)
    pmf = zeros(nbins)

    for b in 1:length(bins)

    bin_map = bins[3] .== bin_grid

    pmf[bin_map] .+= freq_table[b,2]

    return(bins,pmf)

end
#https://www.juliabloggers.com/binning-your-data-with-julia/
bar(reshape(a_grid[1:30],:,1),reshape(sum(D,dims=1)[1:30],:,1))





### Comparative static on β

β_grid = linspace(0.984,0.996,30)

avgmpc_list = zeros(size(β_grid));
avgA_list = zeros(size(β_grid));

for k in 1:length(β_grid)

    tmp = β_grid[k]
    println("Running SS for β = $tmp...\n")
    # Backward iteration
    Va, a, c, normdiff,ss_error= policy_ss(Pi, a_grid, s, Y, r, β_grid[k], eis)
    # Forward iteration
    D = distribution_ss(Pi, a, a_grid)
    # Compute aggregates
    mpcs, avgmpc_list[k] = mpc(c,a_grid)
    avgA_list[k] = sum(broadcast(*,D,reshape(a_grid,1,:)))

end


plot(β_grid, avgmpc_list)
plot(β_grid, avgA_list)
