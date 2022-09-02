
using Plots
using Distributions
using StatsBase
using NLsolve
using LinearAlgebra
using LinearSolve


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



a_grid = discretize_assets(0.0, 10000.0, 50)


### 1B) Income discretization --------------------------

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



Pi = rouwenhorst_Pi(10,0.6)

# First method (Julia fixed point)
n = size(Pi)[1];
guess = fill(1/n, n, 1);
f(pi) = Pi*pi;
sol = fixedpoint(f, guess)
println("Fixed point = $(sol.zero), and |f(x) - x| = $(norm(f(sol.zero) - sol.zero)) in " *
        "$(sol.iterations) iterations")

# Second method (user defined fixed point)
function stationary_markov(Pi, tol=1E-14, maxiter = 10000)
    # start with uniform distribution over all states
    n = size(Pi)[1]
    pi_old = fill(1/n, n, 1);
    normdiff = Inf
    iter = 1
    
    # update distribution using Pi until successive iterations differ by less than tol
    while normdiff > tol && iter <= maxiter
        pi_new = Pi*pi_old # the f(v) map
        normdiff = norm(pi_new - pi_old)
        # replace and continue
        pi_old = pi_new
        iter = iter + 1
    end
    return (pi_old, normdiff, iter) # returns a tuple
end

#pi_star, error, iter = stationary_markov(Pi)

#println("Fixed point = $pi_star, and |f(x) - x| = $error in $iter iterations")



# Third method (QR decomposition: https://stephens999.github.io/fiveMinuteStats/stationary_distribution.html)
A =  [Matrix(1.0I, 10, 10)- Pi;
  ones(1,10)]
b = [zeros(10,1); 1]

Q,R = qr(A)
b_prime = Q'*b 
prob = LinearProblem(R, b_prime)
sol2 = solve(prob)
sol2.u




function discretize_income(rho, sigma, n_s)
    # choose inner-switching probability p to match persistence rho
    p = (1+rho)/2
    
    # start with states from 0 to n_s-1, scale by alpha to match standard deviation sigma
    s = range(0,n_s-1,step=1)
    alpha = 2*sigma/sqrt(n_s-1)
    s = alpha*s
    
    # obtain Markov transition matrix Pi and its stationary distribution
    Pi = rouwenhorst_Pi(n_s, p)
    pi_star,error,iter = stationary_markov(Pi)
    
    # s is log income, get income y and scale so that mean is 1
    y = exp.(s)
    y /= dot(pi_star, y)
    
    return y, pi_star, Pi
end

y, pi_star, Pi = discretize_income(0.975, 0.7, 7)



### (2) Backward iteration ----------------------
beta = 0.99

n_s = 7
n_a = 100

Va = randn(n_s,n_a)
Pi = rouwenhorst_Pi(n_s,0.975)

Wa = beta*Pi*Va