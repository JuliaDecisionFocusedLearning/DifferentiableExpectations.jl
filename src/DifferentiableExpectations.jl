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
using DensityInterface: logdensityof
using Distributions: Distribution, gradlogpdf
using DocStringExtensions
using LinearAlgebra: dot
using OhMyThreads: tmap, treduce, tmapreduce
using Random: Random, AbstractRNG, default_rng
using Statistics: Statistics, mean
using StatsBase: StatsBase

include("utils.jl")
include("abstract.jl")
include("reinforce.jl")
include("reparametrization.jl")
include("distribution.jl")
include("pushforward.jl")

export DifferentiableExpectation
export REINFORCE
export FixedAtomsProbabilityDistribution

end # module DifferentiableExpectations
