"""
    DifferentiableExpectation{t}

Abstract supertype for differentiable parametric expectations `E : θ -> 𝔼[f(X)]` where `X ∼ p(θ)`, whose value and derivative are approximated with Monte-Carlo averages.

# Subtypes

  - [`Reinforce`](@ref)
  - [`Reparametrization`](@ref)

# Calling behavior

    (E::DifferentiableExpectation)(θ...; kwargs...)

Return a Monte-Carlo average `(1/S) ∑f(xᵢ)` where the `xᵢ ∼ p(θ)` are iid samples.

# Type parameters

  - `threaded::Bool`: specifies whether the sampling should be performed in parallel

# Required fields

  - `f`: The function applied inside the expectation.
  - `dist_constructor`: The constructor of the probability distribution.
  - `rng::AbstractRNG`: The random number generator.
  - `nb_samples::Integer`: The number of Monte-Carlo samples.
  - `seed`::Union{Nothing,Integer}: The seed for the random number generator, reset before each call. Set to `nothing` for no seeding.

The field `dist_constructor` must be a callable such that `dist_constructor(θ...)` generates an object `dist` that corresponds to `p(θ)`.
The resulting object `dist` needs to satisfy:

  - the [Random API](https://docs.julialang.org/en/v1/stdlib/Random/#Hooking-into-the-Random-API) for sampling with `rand(rng, dist)`
  - the [DensityInterface.jl API](https://github.com/JuliaMath/DensityInterface.jl) for loglikelihoods with `logdensityof(dist, x)`
"""
abstract type DifferentiableExpectation{t} end

is_threaded(::DifferentiableExpectation{t}) where {t} = Val(t)

"""
    empirical_predistribution(E::DifferentiableExpectation, θ...)

Return a uniform [`FixedAtomsProbabilityDistribution`](@ref) over `{x₁, ..., xₛ}`, where the `xᵢ ∼ p(θ)` are iid samples.
"""
function empirical_predistribution(E::DifferentiableExpectation, θ...)
    (; dist_constructor, rng, nb_samples, seed) = E
    dist = dist_constructor(θ...)
    isnothing(seed) || seed!(rng, seed)
    xs = maybe_eachcol(rand(rng, dist, nb_samples))
    xdist = FixedAtomsProbabilityDistribution(xs; threaded=unval(is_threaded(E)))
    return xdist
end

"""
    empirical_distribution(E::DifferentiableExpectation, θ...; kwargs...)

Return a uniform [`FixedAtomsProbabilityDistribution`](@ref) over `{f(x₁), ..., f(xₛ)}`, where the `xᵢ ∼ p(θ)` are iid samples.
"""
function empirical_distribution(E::DifferentiableExpectation, θ...; kwargs...)
    xdist = empirical_predistribution(E, θ...)
    fk = FixKwargs(E.f, kwargs)
    ydist = map(fk, xdist)
    return ydist
end

function (E::DifferentiableExpectation)(θ...; kwargs...)
    ydist = empirical_distribution(E, θ...; kwargs...)
    return mean(ydist)
end
