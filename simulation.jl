# This is the Julia script to run some small simulations complementary to the Stats Learning Project

# The packages needed

using Plots, LinearAlgebra, StatsBase, Random, Distributions
TITLE = font(20, "Times");
TICKS = font(14, "Times");
GUIDE = font(16, "Times");
LEGEN = font(14, "Times");
cd("G:\\My Drive\\tmp\\project");
include("slfunctions.jl");

using JLD
nmean, nstd, nevent = load("result.jld", "nmean", "nstd", "nevent");



## Let's plot the set K in a 2-d case
Random.seed!(3414811267265127); # for reproducibility
M = randn(2,2);
M = M' * M; # for positive definiteness
ρ = maximum(diag(M));

t = 0:0.001:2*pi;
x = cos.(t);
y = sin.(t);
g = convert(Array{Float64, 2}, M^(-0.5)) * vcat(x', y');
u, v = g[1,:], g[2,:];
normgrid = abs.(u) .+ abs.(v);

r = [0; 0.25; 0.5; 1.0; 2.0; 4.0; 8.0; 16.0];
l1 = r / (2*ρ*sqrt(log(2)));
c = [:black; :dodgerblue; :green3; :orange; :purple; :magenta; :blue; :red];


#p = scatter(x, y, markersize=0.25, markerstrokewidth=0.0, markercolor=:gray80, markeralpha=0.05, label="");
p = scatter(u, v, markersize=5.0, markerstrokewidth=0.0, markercolor=:gray50, markeralpha=0.05, label="Ellipse");

for i = 2:length(r)
    xgrid = 0:0.01:l1[i];
    ygrid = l1[i] .- xgrid;
    scatter!([xgrid; -xgrid; -xgrid; xgrid], [ygrid; ygrid; -ygrid; -ygrid], markersize=0.1; markerstrokewidth=0.0, markercolor=c[i], markeralpha=0.3, label="r = " * string(r[i]));
    idx = (normgrid .> l1[i-1]) .& (normgrid .< l1[i]);
    length(findall(idx)) == 0 ? continue : nothing;
    scatter!(u[idx], v[idx], markersize=2.0, markercolor=c[i], markeralpha=1.0, markerstrokewidth=0.0, label="");
end

xlims!(-4, 4);
ylims!(-4, 4);
plot!(size=(800, 800), dpi=600,xtickfont=TICKS, ytickfont=TICKS, titlefont=TITLE, guidefont=GUIDE, legendfont=LEGEN)
savefig(p, "kset.png");





## Lets now explore the event of A using r = 4,8,16; and explore the effect of n with 50000 simulated X
# use a multidimensional case where d > n

d = 500;
Random.seed!(1 + 3414811267265127); # for reproducibility
M = randn(d, d);
M = M' * M;
ρ = maximum(diag(M));
xdims = collect(50:10:250);
N = 1000;
r = 0.25 * 2 .^ collect(0:16);
#nevents = zeros(length(xdims), 3);
nmean = zeros(length(xdims), length(r));
nstd = zeros(length(xdims),  length(r));
nevent = zeros(length(xdims),  length(r));
mvgaussian = sampler(MvNormal(zeros(d), M)); # for constructing X

# need to sample from uniform the d-dim ellipse surface from metropolis hastings
# which is essentially the g here

##------------ metropolis hastings (overnight, 45555 seconds)
#numsample = 25000;
#@time g = MHEllipse(M, numsample+20; k=300.0, stepsize=2.5e-7)[:, end-numsample+1:end]; # slow due to curse of dimensionality
##----------------------------------------------
normgrid = sum(abs.(g), dims=1)[1, :];

# calculate the norm of min[sqrt(||Xθ||^2/n)] and the probability that min[sqrt(||Xθ||^2/n)] < 0.5-2r
for i = 1: length(r) # for a given radius
    println("- ru = $(r[i]) ... ");
    for j = 1:length(xdims) # for a given dimension
        println("-- n = $(xdims[j]) ...");
        #event = 0;
        l1norm = r[i] / (2*ρ*sqrt(log(d)/xdims[j]));
        idx = i == 1 ? (normgrid .< l1norm) : (normgrid .> 0.5*l1norm) .& (normgrid .< l1norm); # the K set
        if length(findall(idx)) == 0
            @warn("--- K set is empty ...");
            continue;
        else
            @info("--- K set has $(length(findall(idx))) numerical samples ... ");
        end

        # try N different X on all the possible samples in the K set
        s = zeros(N);
        for k = 1:N
            X = rand(mvgaussian, xdims[j])';
            xt = X * g[:, idx]; # operation on k set
            stats = sqrt.(mean(xt.^2, dims=1));
            s[k] = minimum(stats);
        end

        # collect the stats
        nmean[j, i] = mean(s);
        nstd[j, i] = std(s);
        nevent[j, i] = length(findall(s .< 0.5 - 2*r[i]));
    end
end

## Plotting the results a
p = plot(size=(1200, 800), dpi=600, xtickfont=TICKS, ytickfont=TICKS, titlefont=TITLE, guidefont=GUIDE, legendfont=LEGEN);
for j = 1:length(xdims)
    ix = findall(nmean[j, :] .!= 0.0);
    length(ix) == 0 ? continue : nothing;
    plot!(r[ix], nmean[j, ix], ribbon=nstd[j, ix], xscale=:log10, linewidth=3.0, fillalpha=0.1, label="n = $(xdims[j])");
end
xlims!(64, 2e5);
xlabel!("ru");
ylabel!("inf norm");
savefig(p, "simulation.png")









## Just plots showing good sampling on the 2-d case and 3-d case, 4-d is beyond human brain
d = 2
Random.seed!(1 + 3414811267265127); # for reproducibility
M = randn(d, d);
M = M' * M;
@time s = MHEllipse(M, 10000; stepsize=0.1); # it's very effecient 0.2s
p = scatter(s[1,:], s[2,:], markersize=1.0, markerstrokewidth=0.0, markeralpha=0.5, label="");
plot!(size=(800, 800), dpi=600,xtickfont=TICKS, ytickfont=TICKS, titlefont=TITLE, guidefont=GUIDE, legendfont=LEGEN)
savefig(p, "mh2d.png");


d = 3
Random.seed!(1 + 3414811267265127); # for reproducibility
M = randn(d, d);
M = M' * M;
@time s = MHEllipse(M, 10000; stepsize=0.1); # it's very effecient 0.2s
p = scatter(s[1,:], s[2,:], s[3,:], markersize=1.0, markerstrokewidth=0.0, markeralpha=0.5, label="");
plot!(size=(800, 800), dpi=600,xtickfont=TICKS, ytickfont=TICKS, titlefont=TITLE, guidefont=GUIDE, legendfont=LEGEN)
savefig(p, "mh3d.png");


d = 500;
Random.seed!(1 + 3414811267265127); # for reproducibility
M = randn(d, d);
M = M' * M;
numsample = 1000;
@time g = MHEllipse(M, numsample; k=300.0, stepsize=2.5e-7);
dist = zeros(numsample);
for i = 1:numsample
    dist[i] = g[:, i]' * M * g[:, i];
end
p = plot(dist, linewidth=2.0, linealpha=0.8);
xlims!(15, 1000);
ylims!(0.8, 1.2);
ylabel!("Distance to Surface");
xlabel!("Sample #")
plot!(size=(1000, 600), dpi=600,xtickfont=TICKS, ytickfont=TICKS, titlefont=TITLE, guidefont=GUIDE, legendfont=LEGEN);
savefig(p, "mhhd.png");
