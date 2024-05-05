"""
    AbstractExpectation{threaded}

Abstract supertype for expectation wrappers.

The type parameter `threaded` is a `Bool` stating whether or not the Monte-Carlo samples should be computed in parallel.

Implementing subtypes must have the following fields:

- `rng::AbstractRNG`: the random number generator
- `nb_samples::Integer`: the number of samples to draw
"""
abstract type AbstractExpectation{threaded} end
