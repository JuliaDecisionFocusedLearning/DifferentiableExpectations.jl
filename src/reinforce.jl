"""
    Reinforce{threaded} <: DifferentiableExpectation{threaded}

Differentiable parametric expectation `F : θ -> 𝔼[f(X)]` where `X ∼ p(θ)` using the REINFORCE (or score function) gradient estimator:
```
∂F(θ) = 𝔼[f(X) ∇₂logp(X,θ)ᵀ]
```

# Example

```jldoctest
using DifferentiableExpectations, Distributions, Zygote

F = Reinforce(exp, Normal; nb_samples=10^5)
F_true(μ, σ) = mean(LogNormal(μ, σ))

μ, σ = 0.5, 1,0
∇F, ∇F_true = gradient(F, μ, σ), gradient(F_true, μ, σ)
isapprox(collect(∇F), collect(∇F_true); rtol=1e-1)

# output

true
```

# Constructor

    Reinforce(
        f,
        dist_constructor,
        dist_logdensity_grad=nothing;
        rng=Random.default_rng(),
        nb_samples=1,
        threaded=false,
        seed=nothing
    )

# Fields

$(TYPEDFIELDS)

# See also

- [`DifferentiableExpectation`](@ref)
"""
struct Reinforce{threaded,variance_reduction,F,D,G,R<:AbstractRNG,S<:Union{Int,Nothing}} <:
       DifferentiableExpectation{threaded}
    "function applied inside the expectation"
    f::F
    "constructor of the probability distribution `(θ...) -> p(θ)`"
    dist_constructor::D
    "either `nothing` or a parameter gradient callable `(x, θ...) -> ∇₂logp(x, θ)`"
    dist_logdensity_grad::G
    "random number generator"
    rng::R
    "number of Monte-Carlo samples"
    nb_samples::Int
    "seed for the random number generator, reset before each call. Set to `nothing` for no seeding."
    seed::S
end

function Reinforce(
    f::F,
    dist_constructor::D,
    dist_logdensity_grad::G=nothing;
    rng::R=default_rng(),
    nb_samples=1,
    threaded=false,
    variance_reduction=true,
    seed::S=nothing,
) where {F,D,G,R,S}
    return Reinforce{threaded,variance_reduction,F,D,G,R,S}(
        f, dist_constructor, dist_logdensity_grad, rng, nb_samples, seed
    )
end

function dist_logdensity_grad(
    rc::RuleConfig, F::Reinforce{threaded}, x, θ...
) where {threaded}
    (; dist_constructor, dist_logdensity_grad) = F
    if !isnothing(dist_logdensity_grad)
        dθ = dist_logdensity_grad(x, θ...)
    else
        _logdensity_partial(_θ...) = logdensityof(dist_constructor(_θ...), x)
        l, pullback = rrule_via_ad(rc, _logdensity_partial, θ...)
        dθ = Base.tail(pullback(one(l)))
    end
    return dθ
end

function ChainRulesCore.rrule(
    rc::RuleConfig, F::Reinforce{threaded,variance_reduction}, θ...; kwargs...
) where {threaded,variance_reduction}
    project_θ = ProjectTo(θ)

    (; nb_samples) = F
    xs = presamples(F, θ...)
    ys = samples_from_presamples(F, xs; kwargs...)
    y = tmean_or_mean(F, ys)

    _dist_logdensity_grad_partial(x) = dist_logdensity_grad(rc, F, x, θ...)
    gs = tmap_or_map(F, _dist_logdensity_grad_partial, xs)

    ys_with_baseline = if (variance_reduction && nb_samples > 1)
        tmap_or_map(F, yi -> yi .- y, ys)
    else
        ys
    end
    K = nb_samples - (variance_reduction && nb_samples > 1)

    function pullback_Reinforce(dy_thunked)
        dy = unthunk(dy_thunked)
        dF = @not_implemented(
            "The fields of the `Reinforce` object are considered constant."
        )
        _single_sample_pullback(g, y) = g .* dot(y, dy)
        dθ =
            tmapreduce_or_mapreduce(F, _single_sample_pullback, .+, gs, ys_with_baseline) ./
            K
        dθ_proj = project_θ(dθ)
        return (dF, dθ_proj...)
    end

    return y, pullback_Reinforce
end
