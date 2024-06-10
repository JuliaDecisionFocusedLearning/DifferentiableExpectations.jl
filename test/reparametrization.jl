using DifferentiableExpectations: reparametrize
using Distributions
using StableRNGs
using Test

rng = StableRNG(63)

@testset "Univariate Normal" begin
    dist = Normal(2.0, 1.0)
    transformed_dist = reparametrize(dist)
    @test mean([rand(rng, transformed_dist) for _ in 1:(10^4)]) ≈ mean(dist) rtol = 1e-1
    @test std([rand(rng, transformed_dist) for _ in 1:(10^4)]) ≈ std(dist) rtol = 1e-1
end
