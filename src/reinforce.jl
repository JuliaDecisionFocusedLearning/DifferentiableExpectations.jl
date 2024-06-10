"""
    REINFORCE{threaded} <: DifferentiableExpectation{threaded}

Differentiable parametric expectation `F : θ -> 𝔼[f(X)]` where `X ∼ p(θ)` using the REINFORCE (or score function) gradient estimator:
```
∂F(θ) = 𝔼[f(X) ∇₁logp(θ, x)ᵀ]
```

# Constructor

    REINFORCE(
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
struct REINFORCE{threaded,F,D,G,R<:AbstractRNG} <: DifferentiableExpectation{threaded}
    f::F
    dist_constructor::D
    dist_logdensity_grad::G
    rng::R
    nb_samples::Int
end

function REINFORCE(
    f::F,
    dist_constructor::D,
    dist_logdensity_grad::G=nothing;
    rng::R=default_rng(),
    nb_samples=1,
    threaded=false,
) where {F,D,G,R}
    return REINFORCE{threaded,F,D,G,R}(
        f, dist_constructor, dist_logdensity_grad, rng, nb_samples
    )
end

function logdensity_grad(rc::RuleConfig, F::REINFORCE{threaded}, x, θ...) where {threaded}
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
    _logdensity_grad_partial(x) = logdensity_grad(rc, F, x, θ...)
    if threaded
        return tmap(_logdensity_grad_partial, xs)
    else
        return map(_logdensity_grad_partial, xs)
    end
end

function logdensity_grads_from_presamples(
    rc::RuleConfig,
    F::DifferentiableExpectation{threaded},
    xs::AbstractMatrix,
    θ...;
    kwargs...,
) where {threaded}
    _logdensity_grad_partial(x) = logdensity_grad(rc, F, x, θ...)
    if threaded
        return tmap(_logdensity_grad_partial, eachcol(xs))
    else
        return map(_logdensity_grad_partial, eachcol(xs))
    end
end

function ChainRulesCore.rrule(
    rc::RuleConfig, F::REINFORCE{threaded}, θ...; kwargs...
) where {threaded}
    project_θ = ProjectTo(θ)

    (; nb_samples) = F
    xs = presamples(F, θ...)
    ys = samples_from_presamples(F, xs; kwargs...)
    gs = logdensity_grads_from_presamples(rc, F, xs, θ...)

    function REINFORCE_pullback(dy_thunked)
        dy = unthunk(dy_thunked)
        dF = @not_implemented(
            "The fields of the `REINFORCE` object are considered constant."
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
    return y, REINFORCE_pullback
end
