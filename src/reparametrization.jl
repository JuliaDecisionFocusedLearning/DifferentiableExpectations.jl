"""
    TransformedDistribution

Represent the probability distribution `p` of a random variable `X ∼ p` with a transformation `X = T(Z)` where `Z ∼ q`.

# Fields

$(TYPEDFIELDS)
"""
struct TransformedDistribution{D,T}
    "the distribution `q` that gets transformed into `p`"
    base_dist::D
    "the transformation function `T`"
    transformation::T
end

"""
    rand(rng, dist::TransformedDistribution)

Sample from `dist` by applying `dist.transformation` to `dist.base_dist`.
"""
function Random.rand(rng::AbstractRNG, dist::TransformedDistribution)
    (; base_dist, transformation) = dist
    return transformation(rand(rng, base_dist))
end

"""
    reparametrize(dist)

Turn a probability distribution `p` into a [`TransformedDistribution`](@ref) `(q, T)` such that the new distribution `q` does not depend on the parameters of `p`.
These parameters are encoded (closed over) in the transformation function `T`.
"""
function reparametrize end

function reparametrize(dist::Normal{T}) where {T}
    base_dist = Normal(zero(T), one(T))
    μ, σ = mean(dist), std(dist)
    transformation(z) = μ + σ * z
    return TransformedDistribution(base_dist, transformation)
end

function reparametrize(dist::MvNormal{T}) where {T}
    n = length(dist)
    base_dist = MvNormal(fill(zero(T), n), Diagonal(fill(one(T), n)))
    μ, Σ = mean(dist), cov(dist)
    C = cholesky(Σ)
    transformation(z) = μ .+ C.L * z
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

E = Reparametrization(exp, Normal; nb_samples=10^4)
E_true(μ, σ) = mean(LogNormal(μ, σ))

μ, σ = 0.5, 1,0
∇E, ∇E_true = gradient(E, μ, σ), gradient(E_true, μ, σ)
isapprox(collect(∇E), collect(∇E_true); rtol=1e-1)

# output

true
```

# Constructor

    Reparametrization(
        f,
        dist_constructor,
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
struct Reparametrization{t,F,D,R<:AbstractRNG,S<:Union{Int,Nothing}} <:
       DifferentiableExpectation{t}
    "function applied inside the expectation"
    f::F
    "constructor of the probability distribution `(θ...) -> p(θ)`"
    dist_constructor::D
    "random number generator"
    rng::R
    "number of Monte-Carlo samples"
    nb_samples::Int
    "seed for the random number generator, reset before each call. Set to `nothing` for no seeding."
    seed::S
end

function Reparametrization(
    f::F,
    dist_constructor::D;
    rng::R=default_rng(),
    nb_samples=1,
    threaded=false,
    seed::S=nothing,
) where {F,D,R,S}
    return Reparametrization{threaded,F,D,R,S}(f, dist_constructor, rng, nb_samples, seed)
end

function ChainRulesCore.rrule(rc::RuleConfig, E::Reparametrization, θ...; kwargs...)
    project_θ = ProjectTo(θ)

    (; f, dist_constructor, rng, nb_samples) = E
    dist = dist_constructor(θ...)
    transformed_dist = reparametrize(dist)
    zs = maybe_eachcol(rand(rng, transformed_dist.base_dist, nb_samples))
    zdist = FixedAtomsProbabilityDistribution(zs; threaded=unval(is_threaded(E)))
    xdist = map(transformed_dist.transformation, zdist)
    fk = FixKwargs(f, kwargs)
    ydist = map(fk, xdist)
    y = mean(ydist)

    function h(zᵢ, θ)
        transformed_dist = reparametrize(dist_constructor(θ...))
        return f(transformed_dist.transformation(zᵢ); kwargs...)
    end

    function pullback_Reparametrization(Δy_thunked)
        Δy = unthunk(Δy_thunked)
        ΔE = @not_implemented("The fields of the `Reparametrization` object are constant.")
        function _single_sample_pullback(zᵢ)
            _, pb = rrule_via_ad(rc, h, zᵢ, θ)
            _, _, Δθ = pb(Δy)
            return Δθ
        end
        Δθ = mymapreduce(is_threaded(E), _single_sample_pullback, .+, zs) ./ nb_samples
        Δθ_proj = project_θ(Δθ)
        return (ΔE, Δθ_proj...)
    end

    return y, pullback_Reparametrization
end
