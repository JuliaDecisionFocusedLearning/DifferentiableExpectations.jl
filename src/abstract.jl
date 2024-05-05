"""
    DifferentiableExpectation{threaded,D}

Abstract supertype for differentiable parametric expectations `F : θ -> 𝔼[f(X)]` where `X ∼ p(θ)`, whose value and derivative are approximated with Monte-Carlo averages.

# Type parameters

- `threaded::Bool`: specifies whether the sampling should be performed in parallel (with OhMyThreads.jl)
- `D::Type`: the type of the probability distribution, such that calling `D(θ...)` generates a sampleable object corresponding to `p(θ)`

# Required fields

- `f`: the function applied inside the expectation
- `rng`: the random number generator
- `nb_samples`: the number of Monte-Carlo samples
"""
abstract type DifferentiableExpectation{threaded,D} end

"""
    distribution(F::DifferentiableExpectation, θ...)

Create a sampleable object `p(θ)`.
"""
function distribution(::DifferentiableExpectation{threaded,D}, θ...) where {threaded,D}
    return D(θ...)
end

"""
    (F::DifferentiableExpectation)(θ...)

Return a Monte-Carlo average `(1/s) ∑f(xᵢ)` where the `xᵢ ∼ p(θ)` are iid samples.
"""
function (F::DifferentiableExpectation{threaded})(θ...) where {threaded}
    dist = distribution(F, θ...)
    _sample(_) = F.f(rand(F.rng, dist))
    y = if threaded
        tmapmean(_sample, 1:(F.nb_samples))
    else
        mean(_sample, 1:(F.nb_samples))
    end
    return y
end

"""
    pre_samples(F::DifferentiableExpectation, θ...)

Return a vector `[x₁, ..., xₛ]` where the `xᵢ ∼ p(θ)` are iid samples.
"""
function pre_samples(F::DifferentiableExpectation{threaded}, θ...) where {threaded}
    dist = distribution(F, θ...)
    _pre_sample(_) = rand(F.rng, dist)
    xs = if threaded
        tmap(_pre_sample, 1:(F.nb_samples))
    else
        map(_pre_sample, 1:(F.nb_samples))
    end
    return xs
end

"""
    samples(F::DifferentiableExpectation, θ...)

Return a vector `[f(x₁), ..., f(xₛ)]` where the `xᵢ ∼ p(θ)` are iid samples.
"""
function samples(F::DifferentiableExpectation{threaded}, θ...) where {threaded}
    dist = distribution(F, θ...)
    _sample(_) = F.f(rand(F.rng, dist))
    ys = if threaded
        map(_sample, 1:(F.nb_samples))  # TODO: tmap fails here
    else
        map(_sample, 1:(F.nb_samples))
    end
    return ys
end
