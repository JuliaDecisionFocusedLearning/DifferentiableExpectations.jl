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

normal_logdensity_grad(x, θ...) = gradient((_θ...) -> logpdf(Normal(_θ...), x), θ...)

@testset verbose = true "Univariate LogNormal" begin
    μ, σ = 0.5, 1.0
    seed = 63
    true_mean(μ, σ) = mean(LogNormal(μ, σ))
    true_std(μ, σ) = std(LogNormal(μ, σ))
    ∇mean_true = gradient(true_mean, μ, σ)

    @testset verbose = true "Threaded: $threaded" for threaded in (false,)  #
        @testset "$(nameof(typeof(F)))" for F in [
            Reinforce(
                exp_with_kwargs,
                Normal;
                rng=StableRNG(seed),
                nb_samples=10^4,
                threaded=threaded,
                seed=seed,
            ),
            Reinforce(
                exp_with_kwargs,
                Normal,
                normal_logdensity_grad;
                rng=StableRNG(seed),
                nb_samples=10^4,
                threaded=threaded,
                seed=seed,
            ),
            Reparametrization(
                exp_with_kwargs,
                Normal;
                rng=StableRNG(seed),
                nb_samples=10^4,
                threaded=threaded,
                seed=seed,
            ),
        ]
            string(F)

            @test F(μ, σ; correct=true) == F(μ, σ; correct=true)
            @test F.dist_constructor(μ, σ) == Normal(μ, σ)
            @test F(μ, σ; correct=true) ≈ true_mean(μ, σ) rtol = 0.1
            @test std(samples(F, μ, σ; correct=true)) ≈ true_std(μ, σ) rtol = 0.1

            ∇mean_est = gradient((μ, σ) -> F(μ, σ; correct=true), μ, σ)

            @test ∇mean_est[1] ≈ ∇mean_true[1] rtol = 0.2
            @test ∇mean_est[2] ≈ ∇mean_true[2] rtol = 0.2
        end
    end
end;

@testset verbose = true "Multivariate LogNormal" begin
    μ, σ = [2.0, 3.0], [1.0, 0.5]
    true_mean(μ, σ) = mean.(LogNormal.(μ, σ))
    true_std(μ, σ) = std.(LogNormal.(μ, σ))
    ∂mean_true = jacobian(true_mean, μ, σ)

    @testset verbose = true "Threaded: $threaded" for threaded in (false, true)
        @testset "$(nameof(typeof(F)))" for F in [
            Reinforce(
                vec_exp_with_kwargs,
                (μ, σ) -> MvNormal(μ, Diagonal(σ .^ 2));
                rng=StableRNG(63),
                nb_samples=10^5,
                threaded=threaded,
            ),
            # Reparametrization(
            #     vec_exp_with_kwargs,
            #     (μ, σ) -> MvNormal(μ, Diagonal(σ .^ 2));
            #     rng=StableRNG(63),
            #     nb_samples=10^5,
            #     threaded=threaded,
            # ),
        ]
            @test F.dist_constructor(μ, σ) == MvNormal(μ, Diagonal(σ .^ 2))
            @test F(μ, σ; correct=true) ≈ true_mean(μ, σ) rtol = 0.1

            ∂mean_est = jacobian((μ, σ) -> F(μ, σ; correct=true), μ, σ)

            @test ∂mean_est[1] ≈ ∂mean_true[1] rtol = 0.1
            @test ∂mean_est[2] ≈ ∂mean_true[2] rtol = 0.1
        end
    end
end

@testset "Variance reduction" begin
    for seed in 1:10
        rng = StableRNG(seed)
        f(x) = x
        dist_constructor(θ) = MvNormal(θ, I)
        n = 10
        θ = randn(rng, n)
        r = Reinforce(
            f, dist_constructor; rng=rng, nb_samples=100, seed=seed, variance_reduction=true
        )
        r_no_variance_reduction = Reinforce(
            f,
            dist_constructor;
            rng=rng,
            nb_samples=100,
            seed=seed,
            variance_reduction=false,
        )

        J_reduced_variance = jacobian(r, θ)[1]
        J_no_reduced_variance = jacobian(r_no_variance_reduction, θ)[1]
        J_true = Matrix(I, n, n)

        mape(x::AbstractArray, y::AbstractArray) = mean(abs.(x .- y))
        @test mape(J_reduced_variance, J_true) < mape(J_no_reduced_variance, J_true)
    end
end
