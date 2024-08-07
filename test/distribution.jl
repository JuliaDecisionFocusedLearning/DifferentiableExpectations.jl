using ChainRulesCore
using Distributions
using DifferentiableExpectations
using LinearAlgebra
using Random
using StableRNGs
using Statistics
using Test
using Zygote

rng = StableRNG(63)

@test_throws ArgumentError FixedAtomsProbabilityDistribution(Int[], Float64[])
@test_throws DimensionMismatch FixedAtomsProbabilityDistribution([2, 3], [1.0])
@test_throws ArgumentError FixedAtomsProbabilityDistribution([2, 3], [0.4, 0.8])

for threaded in (false, true)
    dist = FixedAtomsProbabilityDistribution([2, 3], [0.4, 0.6]; threaded)

    @test length(dist) == 2

    @test mean(dist) ≈ 2.6
    @test mean(abs2, dist) ≈ 7.0
    @test mean([rand(rng, dist) for _ in 1:(10^5)]) ≈ 2.6 rtol = 0.1
    @test mean(abs2, [rand(rng, dist) for _ in 1:(10^5)]) ≈ 7.0 rtol = 0.1

    @test map(abs2, dist).weights == dist.weights
    @test map(abs2, dist).atoms == [4, 9]

    @test only(gradient(mean, dist)).atoms === nothing
    @test only(gradient(mean, dist)).weights == [2, 3]

    @test last(gradient(mean, abs2, dist)).atoms === nothing
    @test last(gradient(mean, abs2, dist)).weights == [4, 9]
end
