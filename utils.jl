module utils

export rouwenhorst_Pi
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

end