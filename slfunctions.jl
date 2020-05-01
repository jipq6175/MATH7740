

# This is the julia functions for the convenience of running simulations


using LinearAlgebra, StatsBase, Distributions
# generate the samples on the d-dim ellipse using Metropolis Hastings
function MHEllipse(M::Array{Float64, 2}, N::Int64; k::Float64=100.0, stepsize::Float64=0.01)

    n = size(M)[1];
    x0, x1 = zeros(n), randn(n);
    x0[1] = 1/norm(M[:,1]);
    count = 0;
    samples = zeros(n, N);

    while count < N
        x1 = rand(sampler(MvNormal(x0, stepsize * Matrix{Float64}(I, n, n))));
        acceptance = exp(-k*abs(x1' * M * x1 - 1)) / exp(-k*abs(x0' * M * x0 - 1));
        if rand() < acceptance # x1 is better than x0
            count = count + 1;
            count % round(0.01 * N) == 0 ? println("- count = $count ...") : nothing;
            samples[:, count] = x1;
            x0 = x1;
        end
    end

    return samples;
end
