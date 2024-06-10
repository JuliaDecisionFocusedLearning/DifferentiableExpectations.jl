using Distributions
using DifferentiableExpectations
using DifferentiableExpectations: samples
using LinearAlgebra
using Random
using StableRNGs
using Statistics
using Test
using Zygote

exp_with_kwargs(x; correct=false) = correct ? exp(x) : sin(x)
vec_exp_with_kwargs(x; correct=false) = exp_with_kwargs.(x; correct)

@testset "Univariate LogNormal" begin
    for threaded in (false, true)
        F = REINFORCE(
            exp_with_kwargs, Normal; rng=StableRNG(63), nb_samples=10^4, threaded=threaded
        )

        μ, σ = 2.0, 1.0
        true_mean(μ, σ) = mean(LogNormal(μ, σ))
        true_std(μ, σ) = std(LogNormal(μ, σ))

        @test F.dist_constructor(μ, σ) == Normal(μ, σ)
        @test F(μ, σ; correct=true) ≈ true_mean(μ, σ) rtol = 0.1
        @test std(samples(F, μ, σ; correct=true)) ≈ true_std(μ, σ) rtol = 0.1

        ∇mean_est = gradient((μ, σ) -> F(μ, σ; correct=true), μ, σ)
        ∇mean_true = gradient(true_mean, μ, σ)

        @test ∇mean_est[1] ≈ ∇mean_true[1] rtol = 0.2
        @test ∇mean_est[2] ≈ ∇mean_true[2] rtol = 0.2
    end
end

@testset "Multivariate LogNormal" begin
    for threaded in (false, true)
        F = REINFORCE(
            vec_exp_with_kwargs,
            (μ, σ) -> MvNormal(μ, Diagonal(σ .^ 2));
            rng=StableRNG(63),
            nb_samples=10^5,
            threaded=threaded,
        )

        μ, σ = [2.0, 3.0], [1.0, 0.5]
        true_mean(μ, σ) = mean.(LogNormal.(μ, σ))
        true_std(μ, σ) = std.(LogNormal.(μ, σ))

        @test F.dist_constructor(μ, σ) == MvNormal(μ, Diagonal(σ .^ 2))
        @test F(μ, σ; correct=true) ≈ true_mean(μ, σ) rtol = 0.1

        ∂mean_est = jacobian((μ, σ) -> F(μ, σ; correct=true), μ, σ)
        ∂mean_true = jacobian(true_mean, μ, σ)

        @test ∂mean_est[1] ≈ ∂mean_true[1] rtol = 0.1
        @test ∂mean_est[2] ≈ ∂mean_true[2] rtol = 0.1
    end
end
