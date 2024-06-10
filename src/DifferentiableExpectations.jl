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
    @not_implemented,
    rrule_via_ad,
    unthunk
using DensityInterface: logdensityof
using Distributions: Distribution, gradlogpdf
using DocStringExtensions
using LinearAlgebra: dot
using OhMyThreads: tmap, treduce, tmapreduce
using Random: AbstractRNG, default_rng
using Statistics: mean

include("utils.jl")
include("abstract.jl")
include("reinforce.jl")
include("reparametrization.jl")
include("pushforward.jl")

export DifferentiableExpectation
export REINFORCE

end # module DifferentiableExpectations
