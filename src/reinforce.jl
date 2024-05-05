"""
    REINFORCE <: DifferentiableExpectation

Differentiable parametric expectation `F : θ -> 𝔼[f(X)]` where `X ∼ p(θ)` using the REINFORCE (or score function) gradient estimator:
```
∂F(θ) = 𝔼[f(X) ∇₁logp(θ, x)ᵀ]
```

# Constructor

    REINFORCE(; f, dist_type::Type, rng::AbstractRNG, nb_samples::Integer, threaded::Bool)

# Fields

$(TYPEDFIELDS)

# See also

- [`DifferentiableExpectation`](@ref)
"""
struct REINFORCE{threaded,D,F,R<:AbstractRNG} <: DifferentiableExpectation{threaded,D}
    f::F
    rng::R
    nb_samples::Int
end

"""
    REINFORCE(
        ::Type{D}, f;
        rng::AbstractRNG=default_rng(),
        nb_samples::Integer=1,
        threaded::Bool=false
    )

Constructor for [`REINFORCE`](@ref).
"""
function REINFORCE(
    ::Type{D}, f::F; rng::R=default_rng(), nb_samples=1, threaded=false
) where {F,D,R}
    return REINFORCE{threaded,D,F,R}(f, rng, nb_samples)
end

function logdensity_grad(rc::RuleConfig, F::REINFORCE{threaded}, x, θ...) where {threaded}
    _logdensity_partial(_θ...) = logdensityof(distribution(F, _θ...), x)
    l, pullback = rrule_via_ad(rc, _logdensity_partial, θ...)
    dθ = Base.tail(pullback(one(l)))
    return dθ
end

function ChainRulesCore.rrule(rc::RuleConfig, F::REINFORCE{threaded}, θ...) where {threaded}
    (; nb_samples) = F
    _logdensity_grad_partial(x) = logdensity_grad(rc, F, x, θ...)
    xs = pre_samples(F, θ...)
    ys = threaded ? tmap(F.f, xs) : map(F.f, xs)
    gs = threaded ? tmap(_logdensity_grad_partial, xs) : map(_logdensity_grad_partial, xs)
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
        return (dF, dθ...)
    end
    y = threaded ? tmean(ys) : mean(ys)
    return y, REINFORCE_pullback
end
