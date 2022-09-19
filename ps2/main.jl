# PS2 ECOn8861 - Federico Rodari

# --------------------------------------------------------------
# Import packages ----------------------------------------------
#---------------------------------------------------------------
using Plots
using Distributions
using StatsBase
using NLsolve
using LinearAlgebra
using LinearSolve
using Interpolations
using CategoricalArrays
using LaTeXStrings
using Roots
#---------------------------------------------------------------

#---------------------------------------------------------------
# Setup --------------------------------------------------------
#---------------------------------------------------------------

#---------------------------------------------------------------
# Workind directory
cd("$(homedir())\\Dropbox\\Github\\monetary-economics-I")

#---------------------------------------------------------------
# Custom modules import
include("$(homedir())\\Dropbox\\Github\\monetary-economics-I\\modules\\00-utils.jl")
using .utils
include("$(homedir())\\Dropbox\\Github\\monetary-economics-I\\modules\\01-ss_solution.jl")
using .ss_solution

#---------------------------------------------------------------
# Setup parent and children directories
parent_dir = "C:\\Users\\feder\\Dropbox\\Github\\monetary-economics-I\\ps2"
figures_dir = string(parent_dir,"\\figures")
# Create figures folder
if isdir(figures_dir)
else  
    mkdir(figures_dir)
end

#---------------------------------------------------------------
## Global Parameters Setup 
# remember to always have s on the y axis, and a on the x axis
β = 0.988; #1-0.08/4
n_s = 7;
n_a = 500;
amax = 200.0; #10000.0;
amin = 0.0;
ρ,σ = 0.975,0.7;
r = 0.01/4;
Yss = 1;
eis = 1; # log utility implies EIS = 1
ρ_eta,σ_eta = 0.8,0.01;
T = 300; # truncation horizon
# News shock horizon
news_horizon = 10; 
# Horizon comparative static
H = 100:10:1000;
T_max = 50; # horizon to be shown in the plots

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

#---------------------------------------------------------------
# Ss Policy Function
Va_ss, a_ss, c_ss, normdiff,ss_error= policy_ss(Pi, a_grid, s, Yss, r, β, eis);

#---------------------------------------------------------------
# Ergodic Distribution
D_ss = distribution_ss(Pi,a_ss, a_grid); 
# Ss Aggregates
A_ss = dot(D_ss,a_ss);
C_ss = dot(D_ss,c_ss);


#---------------------------------------------------------------
# Sections     -------------------------------------------------
#---------------------------------------------------------------

#---------------------------------------------------------------
# Chose which section to run among:
# 1) mit-shock
# 2) truncation-horizon
# 3) news-shock
# Copy-paste one of the sections and assign it to the variable "section"

section = "news-shock"


if section == "mit-shock"
    print("Running section -shock-\n")

    #---------------------------------------------------------------
    # MIT shock ----------------------------------------------------
    #---------------------------------------------------------------

    #---------------------------------------------------------------
    # Setup Y dynamic
    Y_s = zeros(T)
    #Y_s[1] = Yss

    for s in 1:1:T
        Y_s[s] = Yss +(ρ_eta)^s*σ_eta
    end

    #---------------------------------------------------------------
    # Allocate memory for transitional dynamics matrices
    a = zeros(n_s,n_a,T);
    c = zeros(n_s,n_a,T);

    # Allocate memory for transitional distribution
    D = zeros(n_s,n_a,T);

    # Starting boundary condition
    D[:,:,1] = D_ss;
    # Ending boundary condition
    Va = Va_ss;

    # Iterate from terminal condition
    for t in T:-1:1
        Va, a[:,:,t], c[:,:,t]= backward_iteration(Va, Pi, a_grid, s, r, β, Y_s[t], eis)
    end

    # Iterate from starting condition
    for t in 2:1:T

    D[:,:,t] = forward_iteration(D[:,:,t-1],Pi,a[:,:,t],a_grid)

    end

    # Compute Aggregates
    A = zeros(T);
    C = zeros(T);
    for t in 1:1:T

        A[t] = dot(D[:,:,t], a[:,:,t]);
        C[t] = dot(D[:,:,t], c[:,:,t]);
    end

    # Compute IRFs
    C_irf = 100*(C.-C_ss)./C_ss;
    A_irf = 100*(A.-A_ss)./A_ss;

    # Plot IRFs
    plot(C_irf[1:50], label = L"\textrm{C}",
                  xlabel = L"\textrm{t} + \textrm{h}",
                  ylabel = L"\%\Delta_{ss}",
                  color = "red")
                  
    plot!(A_irf[1:50], label = L"\textrm{A}", color = "blue")

    #save plot
    savefig(string(figures_dir, "\\$section",".pdf"))

elseif section == "truncation-horizon"
    print("Running section -truncation-horizon-\n")

    #---------------------------------------------------------------
    # Play with truncation horizon T -------------------------------
    #---------------------------------------------------------------
    
    #---------------------------------------------------------------
    # Setup
    A_irf_H = zeros(T_max,length(H));
    C_irf_H = zeros(T_max,length(H));


    for T in eachindex(H)

        print("Iteration $T...\n")
        Y_s = zeros(H[T])
        Y_s[1] = Yss

        for s in 2:1:H[T]
            Y_s[s] = Yss +(ρ_eta)^s*σ_eta
        end

        #---------------------------------------------------------------
        # Allocate memory for transitional dynamics matrices
        a = zeros(n_s,n_a,H[T]);
        c = zeros(n_s,n_a,H[T]);

        #---------------------------------------------------------------
        # Allocate memory for D along the transition path
        D = zeros(n_s,n_a,H[T]);
        D[:,:,1] = D_ss;

        # Initialize Va
        Va = Va_ss;

        # Transitional policy function
        for t in H[T]:-1:1
            Va, a[:,:,t], c[:,:,t]= backward_iteration(Va, Pi, a_grid, s, r, β, Y_s[t], eis)
        end

        # Transitional ergodic distributions
        for t in 2:1:H[T]

        D[:,:,t] = forward_iteration(D[:,:,t-1],Pi,a[:,:,t],a_grid)

        end

        # Define aggregates
        A = zeros(H[T])
        C = zeros(H[T])
        for t in 1:1:H[T]

            A[t] = dot(D[:,:,t], a[:,:,t]);
            C[t] = dot(D[:,:,t], c[:,:,t]);
        end

        # Store IRFs
        C_irf_H[:,T] = 100*(C[1:T_max].-C_ss)./C_ss
        A_irf_H[:,T] = 100*(A[1:T_max].-A_ss)./A_ss

    end # end of loop over horizon truncation
 
    # Check impact by looking at variance across different horizons
    if sum(var(C_irf_H,dims = 2) .< 1E-10) + sum(var(A_irf_H,dims = 2) .< 1E-10) == 2*T_max
        print("Truncation at T has no impact.\n")
    else
        print("Truncation at T has impact.\n")
    end
    


elseif section == "news-shock"
    print("Running section -news-shock-\n")

    #---------------------------------------------------------------
    # News shock ---------------------------------------------------
    #---------------------------------------------------------------

    #---------------------------------------------------------------
    # Setup Y dynamic
    Y_s = zeros(T)
    Y_s[1:(news_horizon-1)] .= Yss

    for s in news_horizon:1:T
        Y_s[s] = Yss +(ρ_eta)^(s-news_horizon)*σ_eta
    end
    #---------------------------------------------------------------
    # Allocate memory for transitional dynamics matrices
    a = zeros(n_s,n_a,T);
    c = zeros(n_s,n_a,T);

    # Allocate memory for transitional distribution
    D = zeros(n_s,n_a,T);

    # Starting boundary condition
    D[:,:,1] = D_ss;
    # Ending boundary condition
    Va = Va_ss;

    # Iterate from terminal condition
    for t in T:-1:1
        Va, a[:,:,t], c[:,:,t]= backward_iteration(Va, Pi, a_grid, s, r, β, Y_s[t], eis)
    end

    # Iterate from starting condition
    for t in 2:1:T

    D[:,:,t] = forward_iteration(D[:,:,t-1],Pi,a[:,:,t],a_grid)

    end

    # Compute Aggregates
    A = zeros(T)
    C = zeros(T)
    for t in 1:1:T

        A[t] = dot(D[:,:,t], a[:,:,t]);
        C[t] = dot(D[:,:,t], c[:,:,t]);
    end

    # Compute IRFs
    C_irf = 100*(C.-C_ss)./C_ss;
    A_irf = 100*(A.-A_ss)./A_ss;
    
    plot(C_irf[1:50], label = L"\textrm{C}",
                  xlabel = L"\textrm{t} + \textrm{h}",
                  ylabel = L"\%\Delta_{ss}",
                  color = "red")
                  
    plot!(A_irf[1:50], label = L"\textrm{A}", color = "blue")

    #save plot
    savefig(string(figures_dir, "\\$section",".pdf"))


else
    print("No valid section selected. Please check your spell!\n")
end


#---------------------------------------------------------------
# Hugget (1993) GE model ---------------------------------------
#---------------------------------------------------------------

#---------------------------------------------------------------
# Calibration of RHS
B_Y = 1.4*4;
G_Y = 0.18;
τ = r*B_Y + G_Y;
rhs = B_Y/(1-τ)

#---------------------------------------------------------------
# Compute aggregate measures
function ss_aggregate(Pi, a_grid, s, Yss, r, β, eis)

    Va_ss, a_ss, c_ss, normdiff,ss_error =  policy_ss(Pi, a_grid,
                                                      s, Yss, r, β, eis)

    D_ss = distribution_ss(Pi,a_ss, a_grid)

    A = dot(D_ss,a_ss)
    C = dot(D_ss,c_ss)
    return(A,C)
end

#---------------------------------------------------------------
# Error function (LHS-RHS) for market-clearing condition
# Output [1] of policy_ss is A
 err(x) = ss_aggregate(Pi, a_grid, (1-τ)*s, Yss, r, x[1], eis)[1]/((1-τ)*Yss)  - rhs


# Explore a chosen bracket to assess the error function behaviour
β_grid = linspace(0.975,0.995,100)
error = zeros(length(β_grid))
for b in eachindex(β_grid)
    print("iteration $b...\n")
    error[b] = err(β_grid[b]);
end


plot(β_grid,
     error,
     label = L"A-B",
     color = "blue",
     xlabel = L"β",
     ylabel = L"\textrm{Error}")

#save plot
savefig(string(figures_dir, "\\bracketing.pdf"))

#---------------------------------------------------------------
# Find market-clearing β
β_root = find_zero(err, (0.975,  0.995))
round(β_root,digits = 4);


