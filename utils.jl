module utils

export linspace,rouwenhorst_Pi, stationary_markov, closest_index,
       closest_value_and_index, get_lottery


using Plots
using Distributions
using StatsBase
using NLsolve
using LinearAlgebra
using LinearSolve
using Interpolations
using CategoricalArrays

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
function stationary_markov(Pi,n_s)

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

end




# ---------------------------------------------------------
# Young's method
# (https://julienpascal.github.io/post/young_2010/)

# Not my function. 
# Credit to: https://discourse.julialang.org/t/findnearest-function/4143/4
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

# Mass point approximation (scalar)
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
