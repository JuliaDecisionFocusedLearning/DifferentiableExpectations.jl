using Distributions
using DifferentiableExpectations
using Random
using StableRNGs
using Statistics
using Test

e = REINFORCE(; f=exp, dist_type=Normal, rng=StableRNG(63), nb_samples=10^3, threaded=false)
μ, σ = 2.0, 1.0
@test distribution(e, μ, σ) == Normal(μ, σ)
@test e(μ, σ) ≈ mean(LogNormal(μ, σ)) rtol = 0.1
@test std(samples(e, μ, σ)) ≈ std(LogNormal(μ, σ)) rtol = 0.1
