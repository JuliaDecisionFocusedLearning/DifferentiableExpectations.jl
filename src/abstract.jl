"""
    DifferentiableExpectation{threaded}

Abstract supertype for differentiable parametric expectations `F : θ -> 𝔼[f(X)]` where `X ∼ p(θ)`, whose value and derivative are approximated with Monte-Carlo averages.

# Subtypes

  - [`Reinforce`](@ref)
  - [`Reparametrization`](@ref)

# Calling behavior

    (F::DifferentiableExpectation)(θ...; kwargs...)

Return a Monte-Carlo average `(1/s) ∑f(xᵢ)` where the `xᵢ ∼ p(θ)` are iid samples.

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
abstract type DifferentiableExpectation{threaded} end

tmap_or_map(::DifferentiableExpectation{true}, args...) = tmap(args...)
tmap_or_map(::DifferentiableExpectation{false}, args...) = map(args...)

tmapreduce_or_mapreduce(::DifferentiableExpectation{true}, args...) = tmapreduce(args...)
tmapreduce_or_mapreduce(::DifferentiableExpectation{false}, args...) = mapreduce(args...)

tmean_or_mean(::DifferentiableExpectation{true}, args...) = tmean(args...)
tmean_or_mean(::DifferentiableExpectation{false}, args...) = mean(args...)

"""
    presamples(F::DifferentiableExpectation, θ...)

Return a vector `[x₁, ..., xₛ]` or matrix `[x₁ ... xₛ]` where the `xᵢ ∼ p(θ)` are iid samples.
"""
function presamples(F::DifferentiableExpectation, θ...)
    (; dist_constructor, rng, nb_samples, seed) = F
    dist = dist_constructor(θ...)
    isnothing(seed) || seed!(rng, seed)
    xs = maybe_eachcol(rand(rng, dist, nb_samples))
    return xs
end

"""
    samples(F::DifferentiableExpectation, θ...; kwargs...)

Return a vector `[f(x₁), ..., f(xₛ)]` where the `xᵢ ∼ p(θ)` are iid samples.
"""
function samples(F::DifferentiableExpectation{threaded}, θ...; kwargs...) where {threaded}
    xs = presamples(F, θ...)
    return samples_from_presamples(F, xs; kwargs...)
end

function samples_from_presamples(
    F::DifferentiableExpectation{threaded}, xs::AbstractVector; kwargs...
) where {threaded}
    (; f) = F
    fk = FixKwargs(f, kwargs)
    return tmap_or_map(F, fk, xs)
end

function (F::DifferentiableExpectation{threaded})(θ...; kwargs...) where {threaded}
    ys = samples(F, θ...; kwargs...)
    y = tmean_or_mean(F, ys)
    return y
end
