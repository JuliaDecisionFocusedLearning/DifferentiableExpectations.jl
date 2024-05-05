using Distributions
using DifferentiableExpectations
using Random
using StableRNGs
using Statistics
using Test
using Zygote

@testset "Univariate LogNormal" begin
    for threaded in (false, true)
        F = REINFORCE(Normal, exp; rng=StableRNG(63), nb_samples=10^5, threaded=threaded)
        μ, σ = 2.0, 1.0
        true_mean(μ, σ) = mean(LogNormal(μ, σ))
        true_std(μ, σ) = std(LogNormal(μ, σ))

        @test distribution(F, μ, σ) == Normal(μ, σ)
        @test F(μ, σ) ≈ true_mean(μ, σ) rtol = 0.1
        @test std(samples(F, μ, σ)) ≈ true_std(μ, σ) rtol = 0.1

        ∇mean_est = gradient(F, μ, σ)
        ∇mean_true = gradient(true_mean, μ, σ)

        ∇std_est = gradient((_μ, _σ) -> std(samples(F, _μ, _σ)), μ, σ)
        ∇std_true = gradient(true_std, μ, σ)

        for i in 1:2
            @test ∇mean_est[i] ≈ ∇mean_true[i] rtol = 0.2
            @test ∇std_est[i] ≈ ∇std_true[i] rtol = 0.2
        end
    end
end
