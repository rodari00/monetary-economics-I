
#---------------------------------------------------------------
# Federico Rodari (2022)
# Problem Set 1 ECON 8861
#---------------------------------------------------------------


#---------------------------------------------------------------
# Initial Setup ------------------------------------------------
#---------------------------------------------------------------

# Standard modules import
using Plots
using Distributions
using StatsBase
using NLsolve
using LinearAlgebra
using LinearSolve
using Interpolations
using CategoricalArrays
using LaTeXStrings

# Custom modules import
include("utils.jl")
using .utils
include("01-ss_solution.jl")
using .ss_solution

# Setup parent and children directories
parent_dir = "C:\\Users\\feder\\Dropbox\\Github\\monetary-economics-I"
figures_dir = "$parent_dir\\figures"


## Global Parameters Setup ---------------------------
# remember to always have s on the y axis, and a on the x axis
β = 0.988; #1-0.08/4
n_s = 7;
n_a = 500;
amax = 200.0; #10000.0;
amin = 0.0;
ρ,σ = 0.975,0.7;
r = 0.01/4;
Y = 1;
eis = 1; # log utility implies EIS = 1



#---------------------------------------------------------------
# States Discretization ----------------------------------------
#---------------------------------------------------------------

# Generate the grid (double exponential method)
a_grid = discretize_assets(amin, amax, 500);
# Generate productivity grid
s, pi_star, Pi = discretize_productivity(ρ, σ, n_s);

#---------------------------------------------------------------
# Compute S.S. -------------------------------------------------
#---------------------------------------------------------------
Va, a, c, normdiff,ss_error= policy_ss(Pi, a_grid, s, Y, r, β, eis)

#---------------------------------------------------------------
# Plot Policy function c(a)
gr()
ytmp = round(s[1], digits = 2)
policy_fun = plot(a_grid,c[1,:],
                  label = "y = $ytmp",
                  leg = :bottomright,
                  xlabel = L"a",
                  ylabel = L"c(s,a)")
for state in 2:n_s
    ytmp = round(s[state], digits = 2)
    plot!(policy_fun, a_grid, c[state,:],
         label = "y = $ytmp", 
         leg = :bottomright)
end
policy_fun

#save plot
savefig("$figures_dir\\PS1\\policy_function.pdf")


#---------------------------------------------------------------  
# Plot Savings function (a'-a)(a)
ytmp = round(s[1], digits = 2)
savings_fun = plot(a_grid,a[1,:]-a_grid,
                    label = "y = $ytmp",
                    xlabel = L"a",
                    ylabel = L"a'(s,a) - a")
for state in 2:n_s
    ytmp = round(s[state], digits = 2)
    plot!(a_grid, a[state,:]-a_grid, label = "y = $ytmp")
end
savings_fun

#save plot
savefig("$figures_dir\\PS1\\savings_function.pdf")

#---------------------------------------------------------------
# Stationary Distribution
D = distribution_ss(Pi,a, a_grid) # sensitive to starting condition?
# CDF
F_d = cumsum(sum(D, dims = 1), dims = 2)

# Plot the marginal CDF F(a)
plot(a_grid[1:150],reshape(F_d,:,1)[1:150], label = "F(a)")


#-------------------------------------------------------
# Average MPC ------------------------------------------
#-------------------------------------------------------

# Function for average and conditional MPC
function mpc(c,a_grid)

    # Initialize matrix
    mpcs = zeros(size(c))

    # Interior forward difference
    mpcs[:,2:end-1] = (1+r)*((c[:,3:end]- c[:,2:end-1])./reshape((a_grid[3:end] -a_grid[2:end-1]),1,:))

    # Boundary differences
    mpcs[:,1] = (1+r)*((c[:,2]- c[:,1])./(a_grid[2] -a_grid[1]))
    mpcs[:,end] = (1+r)*((c[:,end]- c[:,end-1])./(a_grid[end] -a_grid[end-1]))

    # Enforce unitary MPC at the boundary
    mpcs[a.==a_grid[1]] .= 1

    # Average MPC
    avgmpc = sum(broadcast(*,D,mpcs))

    return(mpcs, avgmpc)
end


mpcs,avgmpc = mpc(c,a_grid)

# Plot conditional MPCs distribution 
ytmp = round(s[1], digits = 2)
mpc_dist = plot(a_grid[1:40],mpcs[1,1:40],
                label = "y = $ytmp",
                xlabel = "a",
                ylabel = "MPC")
for state in 2:n_s
    ytmp = round(s[state], digits = 2)
    plot!(a_grid[1:40], mpcs[state,1:40], label = "y = $ytmp")
end
mpc_dist

savefig("$figures_dir\\PS1\\conditional_mpcs.pdf")


#-------------------------------------------------------
# Asset distribution -----------------------------------
#-------------------------------------------------------

#Average Asset
A_avg = sum(broadcast(*,D,reshape(a_grid,1,:).*(0.25*Y)))

# Plot conditional asset distribution f(a'(s,a)|s)
 ytmp = round(s[1], digits = 2)
 a_dist = plot(a_grid[1:150],cumsum(D[1,1:150])./pi_star[1],
                label = "y = $ytmp",
                xlabel = L"a\; (\textrm{truncated})",
                ylabel = L"F(a)")
 for state in 2:n_s
     ytmp = round(s[state], digits = 2)
     plot!(a_grid[1:150], cumsum(D[state,1:150])./pi_star[state], label = "y = $ytmp")
 end
 a_dist

# Plot histogram of asset distribution (uneven bins)
# https://www.juliabloggers.com/binning-your-data-with-julia/
asset_freq = [reshape(a_grid,:,1) reshape(sum(D,dims=1),:,1)]

function create_histogram(freq_table,nbins)

    bins = cut(freq_table[:,1],nbins)
    grid = unique(bins)
    pmf = zeros(nbins)

    for b in eachindex(bins)

    bin_map = bins[b] .== grid

    pmf[bin_map] .+= freq_table[b,2]
    end
    # create vector grid
    grid_idx = Vector{String}(undef,nbins)
    for idx in eachindex(grid)
      grid_idx[idx] = string(
                       round(
                        parse(Float64,
                                match(r"(\d*.\d+)(?!.*\d)",grid[idx])[1]),
                        digits = 1)
                            )     
    end
    return(grid,grid_idx,pmf)

end

# Select number of bins
nbins = 30

bin_grid, idx, pmf = create_histogram(asset_freq,20)

bar(idx, pmf,
    xlabel = L"a",
    ylabel = L"f(a)",
     legend = false)

savefig("$figures_dir\\PS1\\asset_distribution_histogram.pdf")

#---------------------------------------------------------------
# Comparative static on β --------------------------------------
#---------------------------------------------------------------

# Initialize grid
β_grid = linspace(0.984,0.996,50)

# Initialize vectors
avgmpc_list = zeros(size(β_grid));
avgA_list = zeros(size(β_grid));

for k in eachindex(β_grid)

    tmp = β_grid[k]
    println("Running SS for β = $tmp...\n")
    # Backward iteration
    Va, a, c, normdiff,ss_error= policy_ss(Pi, a_grid, s, Y, r, β_grid[k], eis)
    # Forward iteration
    D = distribution_ss(Pi, a, a_grid)
    # Compute aggregates
    mpcs, avgmpc_list[k] = mpc(c,a_grid)
    avgA_list[k] = sum(broadcast(*,D,reshape(a_grid,1,:).*(0.25*Y)))

end

# Plot resulting policies
gr()
plot(β_grid, avgmpc_list,
 label = L"\overline{MPC}",
  color = :red,
  xlabel = L"\textrm{Discount\;\; Factor\;\beta}",
  ylabel = L"\textrm{Average\;\;MPC}",
  legend = :topleft,
  axis = :left,
  left_margin = 5Plots.mm,
   right_margin = 15Plots.mm)

plot!(twinx(),β_grid, avgA_list,
    label = L"\overline{A}",
    color = "blue",
    ylabel = L"\textrm{Average\;\;Asset}",
    axis = :right,
    xticks = :off,
    box = :on,
    left_margin = 5Plots.mm,
    right_margin = 15Plots.mm)

savefig("$figures_dir\\PS1\\beta_comparative_static.pdf")

