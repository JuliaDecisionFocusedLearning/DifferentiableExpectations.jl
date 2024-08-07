"""
    DifferentiableExpectations

A Julia package for differentiating through expectations with Monte-Carlo estimates.

# Exports

$(EXPORTS)
"""
module DifferentiableExpectations

using ChainRulesCore:
    ChainRulesCore,
    NoTangent,
    ProjectTo,
    RuleConfig,
    Tangent,
    @not_implemented,
    rrule,
    rrule_via_ad,
    unthunk
using Compat: @compat
using DensityInterface: logdensityof
using Distributions: Distribution, MvNormal, Normal
using DocStringExtensions
using LinearAlgebra: Diagonal, cholesky, dot
using OhMyThreads: tmap, treduce, tmapreduce
using Random: Random, AbstractRNG, default_rng, seed!
using Statistics: Statistics, cov, mean, std
using StatsBase: StatsBase

include("utils.jl")
include("distribution.jl")
include("abstract.jl")
include("reinforce.jl")
include("reparametrization.jl")

export DifferentiableExpectation
export Reinforce
export Reparametrization
export FixedAtomsProbabilityDistribution
export empirical_distribution

@compat public atoms, weights

end # module DifferentiableExpectations
