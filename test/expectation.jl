using Distributions
using DifferentiableExpectations
using DifferentiableExpectations: atoms
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
        @testset "$(nameof(typeof(E)))" for E in [
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
            string(E)

            @test E(μ, σ; correct=true) == E(μ, σ; correct=true)
            @test E.dist_constructor(μ, σ) == Normal(μ, σ)
            @test E(μ, σ; correct=true) ≈ true_mean(μ, σ) rtol = 0.1
            @test std(atoms(empirical_distribution(E, μ, σ; correct=true))) ≈ true_std(μ, σ) rtol =
                0.1

            ∇mean_est = gradient((μ, σ) -> E(μ, σ; correct=true), μ, σ)

            @test ∇mean_est[1] ≈ ∇mean_true[1] rtol = 0.2
            @test ∇mean_est[2] ≈ ∇mean_true[2] rtol = 0.2
        end
    end
end;

@testset verbose = true "Multivariate LogNormal" begin
    μ, σ = [2.0, 3.0], [0.2, 0.3]
    true_mean(μ, σ) = mean.(LogNormal.(μ, σ))
    true_std(μ, σ) = std.(LogNormal.(μ, σ))
    ∂mean_true = jacobian(true_mean, μ, σ)

    @testset verbose = true "Threaded: $threaded" for threaded in (false, true)
        @testset "$(nameof(typeof(E)))" for E in [
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
            @test E.dist_constructor(μ, σ) == MvNormal(μ, Diagonal(σ .^ 2))
            @test E(μ, σ; correct=true) ≈ true_mean(μ, σ) rtol = 0.1

            ∂mean_est = jacobian((μ, σ) -> E(μ, σ; correct=true), μ, σ)

            @test ∂mean_est[1] ≈ ∂mean_true[1] rtol = 0.1
            @test ∂mean_est[2] ≈ ∂mean_true[2] rtol = 0.1
        end
    end
end

@testset "Reinforce variance reduction" begin
    μ, σ = 0.5, 1.0
    seed = 63

    r = Reinforce(exp, Normal; nb_samples=100, variance_reduction=true, rng=StableRNG(seed))
    r_no_vr = Reinforce(
        exp, Normal; nb_samples=100, variance_reduction=false, rng=StableRNG(seed)
    )

    grads = [gradient(r, μ, σ) for _ in 1:1000]
    grads_no_vr = [gradient(r_no_vr, μ, σ) for _ in 1:1000]

    @test var(first.(grads)) < var(first.(grads_no_vr))
    @test var(last.(grads)) < var(last.(grads_no_vr))
end

@testset "Reinforce proba dist rule" begin
    μ, σ = 0.5, 1.0
    seed = 63
    r = Reinforce(
        exp, Normal; nb_samples=100, variance_reduction=false, rng=StableRNG(seed), seed=0
    )
    r_split(θ...) = mean(empirical_distribution(r, θ...))
    @test r(μ, σ) == r_split(μ, σ)
    @test gradient(r, μ, σ) == gradient(r_split, μ, σ)

    r = Reinforce(
        exp, Normal; nb_samples=100, variance_reduction=true, rng=StableRNG(seed), seed=0
    )
    r_split(θ...) = mean(empirical_distribution(r, θ...))
    @test r(μ, σ) == r_split(μ, σ)
    @test_broken gradient(r, μ, σ) == gradient(r_split, μ, σ)
end
