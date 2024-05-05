"""
    AbstractExpectation{threaded}

Abstract supertype for expectation wrappers.

The type parameter `threaded` is a `Bool` stating whether the Monte-Carlo samples should be computed in parallel.
"""
abstract type AbstractExpectation{threaded} end

"""
    distribution(e::AbstractExpectation, θ...)

Build the sampling distribution for `e` based on parameters `\theta`.
"""
function distribution end

"""
    (e::AbstractExpectation)(θ...)

Return the Monte-Carlo average of the function represented by `e` over several samples of `distribution(e, θ...)`.
"""
function (e::AbstractExpectation{threaded})(θ...) where {threaded}
    dist = distribution(e, θ...)
    s = if threaded
        tmapreduce(+, 1:(e.nb_samples)) do _
            e.f(rand(e.rng, dist))
        end
    else
        mapreduce(+, 1:(e.nb_samples)) do _
            e.f(rand(e.rng, dist))
        end
    end
    return s / e.nb_samples
end

"""
    samples(e::AbstractExpectation, θ...)

Return the values of the function represented by `e` for several samples of `distribution(e, θ...)`.
"""
function samples(e::AbstractExpectation{threaded}, θ...) where {threaded}
    dist = distribution(e, θ...)
    if threaded
        return tmap(1:(e.nb_samples)) do _
            e.f(rand(e.rng, dist))
        end
    else
        return map(1:(e.nb_samples)) do _
            e.f(rand(e.rng, dist))
        end
    end
end
