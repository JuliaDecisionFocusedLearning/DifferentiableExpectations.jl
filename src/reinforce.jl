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
        dist_gradlogpdf=nothing;
        rng=Random.default_rng(),
        nb_samples=1,
        threaded=false
    )

# Fields

$(TYPEDFIELDS)

# See also

- [`DifferentiableExpectation`](@ref)
"""
struct Reinforce{threaded,F,D,G,R<:AbstractRNG} <: DifferentiableExpectation{threaded}
    f::F
    dist_constructor::D
    dist_logdensity_grad::G
    rng::R
    nb_samples::Int
end

function Base.show(io::IO, rep::Reinforce{threaded}) where {threaded}
    (; f, dist_constructor, dist_logdensity_grad, rng, nb_samples) = rep
    return print(
        io,
        "Reinforce{$threaded}($f, $dist_constructor, $dist_logdensity_grad, $rng, $nb_samples)",
    )
end

function Reinforce(
    f::F,
    dist_constructor::D,
    dist_logdensity_grad::G=nothing;
    rng::R=default_rng(),
    nb_samples=1,
    threaded=false,
) where {F,D,G,R}
    return Reinforce{threaded,F,D,G,R}(
        f, dist_constructor, dist_logdensity_grad, rng, nb_samples
    )
end

function dist_logdensity_grad(
    rc::RuleConfig, F::Reinforce{threaded}, x, θ...
) where {threaded}
    (; dist_constructor, dist_logdensity_grad) = F
    if !isnothing(dist_logdensity_grad)
        dθ = dist_logdensity_grad(θ...)
    else
        # TODO: add Distributions.gradlogpdf
        _logdensity_partial(_θ...) = logdensityof(dist_constructor(_θ...), x)
        l, pullback = rrule_via_ad(rc, _logdensity_partial, θ...)
        dθ = Base.tail(pullback(one(l)))
    end
    return dθ
end

function logdensity_grads_from_presamples(
    rc::RuleConfig,
    F::DifferentiableExpectation{threaded},
    xs::AbstractVector,
    θ...;
    kwargs...,
) where {threaded}
    _dist_logdensity_grad_partial(x) = dist_logdensity_grad(rc, F, x, θ...)
    if threaded
        return tmap(_dist_logdensity_grad_partial, xs)
    else
        return map(_dist_logdensity_grad_partial, xs)
    end
end

function logdensity_grads_from_presamples(
    rc::RuleConfig,
    F::DifferentiableExpectation{threaded},
    xs::AbstractMatrix,
    θ...;
    kwargs...,
) where {threaded}
    _dist_logdensity_grad_partial(x) = dist_logdensity_grad(rc, F, x, θ...)
    if threaded
        return tmap(_dist_logdensity_grad_partial, eachcol(xs))
    else
        return map(_dist_logdensity_grad_partial, eachcol(xs))
    end
end

function ChainRulesCore.rrule(
    rc::RuleConfig, F::Reinforce{threaded}, θ...; kwargs...
) where {threaded}
    project_θ = ProjectTo(θ)

    (; nb_samples) = F
    xs = presamples(F, θ...)
    ys = samples_from_presamples(F, xs; kwargs...)

    gs = logdensity_grads_from_presamples(rc, F, xs, θ...)

    function pullback_Reinforce(dy_thunked)
        dy = unthunk(dy_thunked)
        dF = @not_implemented(
            "The fields of the `Reinforce` object are considered constant."
        )
        _single_sample_pullback(g, y) = g .* dot(y, dy)
        dθ = if threaded
            tmapreduce(_single_sample_pullback, .+, gs, ys) ./ nb_samples
        else
            mapreduce(_single_sample_pullback, .+, gs, ys) ./ nb_samples
        end
        dθ_proj = project_θ(dθ)
        return (dF, dθ_proj...)
    end

    y = threaded ? tmean(ys) : mean(ys)
    return y, pullback_Reinforce
end
