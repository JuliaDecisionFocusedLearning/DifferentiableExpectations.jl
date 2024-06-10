"""
    DifferentiableExpectation{threaded}

Abstract supertype for differentiable parametric expectations `F : θ -> 𝔼[f(X)]` where `X ∼ p(θ)`, whose value and derivative are approximated with Monte-Carlo averages.

# Type parameters

  - `threaded::Bool`: specifies whether the sampling should be performed in parallel (with OhMyThreads.jl)

# Required fields

  - `f`: the function applied inside the expectation
  - `dist_constructor`: the constructor of the probability distribution, such that calling `D(θ...)` generates an object corresponding to `p(θ)`
  - `rng`: the random number generator
  - `nb_samples`: the number of Monte-Carlo samples
"""
abstract type DifferentiableExpectation{threaded} end

"""
    presamples(F::DifferentiableExpectation, θ...)

Return a vector `[x₁, ..., xₛ]` or matrix `[x₁ ... xₛ]` where the `xᵢ ∼ p(θ)` are iid samples.
"""
function presamples(F::DifferentiableExpectation, θ...)
    (; dist_constructor, rng, nb_samples) = F
    dist = dist_constructor(θ...)
    xs = rand(rng, dist, nb_samples)  # TODO: parallelize?
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
    if threaded
        return tmap(fk, xs)
    else
        return map(fk, xs)
    end
end

function samples_from_presamples(
    F::DifferentiableExpectation{threaded}, xs::AbstractMatrix; kwargs...
) where {threaded}
    (; f) = F
    fk = FixKwargs(f, kwargs)
    if threaded
        return tmap(fk, eachcol(xs))
    else
        return map(fk, eachcol(xs))
    end
end

"""
    (F::DifferentiableExpectation)(θ...; kwargs...)

Return a Monte-Carlo average `(1/s) ∑f(xᵢ)` where the `xᵢ ∼ p(θ)` are iid samples.
"""
function (F::DifferentiableExpectation{threaded})(θ...; kwargs...) where {threaded}
    ys = samples(F, θ...; kwargs...)
    y = if threaded
        tmean(ys)
    else
        mean(ys)
    end
    return y
end
