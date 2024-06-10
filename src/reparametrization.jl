struct TransformedDistribution{D,T}
    base_dist::D
    transformation::T
end

function reparametrize(dist::Normal{T}) where {T}
    base_dist = Normal(zero(T), one(T))
    μ, σ = mean(dist), std(dist)
    transformation(z) = μ + σ * z
    return TransformedDistribution(base_dist, transformation)
end

"""
    Reparametrization{threaded} <: DifferentiableExpectation{threaded}

Differentiable parametric expectation `F : θ -> 𝔼[f(X)]` where `X ∼ p(θ)` using the reparametrization (or pathwise) gradient estimator: if `X = g(Z,θ)` where `Z ∼ q` then
```
∂F(θ) = 𝔼_q[∂f(g(Z,θ)) ∂₂g(Z,θ)ᵀ]
```

# Example

```jldoctest
using DifferentiableExpectations, Distributions, Zygote

F = Reparametrization(exp, Normal; nb_samples=10^3)
F_true(μ, σ) = mean(LogNormal(μ, σ))

μ, σ = 0.5, 1,0
∇F, ∇F_true = gradient(F, μ, σ), gradient(F_true, μ, σ)
isapprox(collect(∇F), collect(∇F_true); rtol=1e-1)

# output

true
```

# Constructor

    Reparametrization(
        f,
        dist_constructor,
        rng=Random.default_rng(),
        nb_samples=1,
        threaded=false
    )

# Fields

$(TYPEDFIELDS)

# See also

- [`DifferentiableExpectation`](@ref)
"""
struct Reparametrization{threaded,F,D,R<:AbstractRNG} <: DifferentiableExpectation{threaded}
    f::F
    dist_constructor::D
    rng::R
    nb_samples::Int
end

function Base.show(io::IO, rep::Reparametrization{threaded}) where {threaded}
    (; f, dist_constructor, rng, nb_samples) = rep
    return print(
        io, "Reparametrization{$threaded}($f, $dist_constructor, $rng, $nb_samples)"
    )
end

function Reparametrization(
    f::F, dist_constructor::D; rng::R=default_rng(), nb_samples=1, threaded=false
) where {F,D,R}
    return Reparametrization{threaded,F,D,R}(f, dist_constructor, rng, nb_samples)
end

function ChainRulesCore.rrule(
    rc::RuleConfig, F::Reparametrization{threaded}, θ...; kwargs...
) where {threaded}
    project_θ = ProjectTo(θ)

    (; f, dist_constructor, rng, nb_samples) = F
    dist = dist_constructor(θ...)
    transformed_dist = reparametrize(dist)
    zs = rand(rng, transformed_dist.base_dist, nb_samples)
    xs = transformed_dist.transformation.(zs)
    ys = samples_from_presamples(F, xs; kwargs...)

    function h(z, θ)
        transformed_dist = reparametrize(dist_constructor(θ...))
        return f(transformed_dist.transformation(z); kwargs...)
    end

    function pullback_Reparametrization(dy_thunked)
        dy = unthunk(dy_thunked)
        dF = @not_implemented(
            "The fields of the `Reparametrization` object are considered constant."
        )
        function _single_sample_pullback(z)
            _, pb = rrule_via_ad(rc, h, z, θ)
            _, _, dθ = pb(dy)
            return dθ
        end
        dθ = if threaded
            tmapreduce(_single_sample_pullback, .+, zs) ./ nb_samples
        else
            mapreduce(_single_sample_pullback, .+, zs) ./ nb_samples
        end
        dθ_proj = project_θ(dθ)
        return (dF, dθ_proj...)
    end

    y = threaded ? tmean(ys) : mean(ys)
    return y, pullback_Reparametrization
end
