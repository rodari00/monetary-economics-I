# Find stationary Markov distribution



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

