module DifferentiableExpectations

using ChainRulesCore: ChainRulesCore, NoTangent, RuleConfig, rrule_via_ad
using Distributions: Distribution
using DocStringExtensions
using OhMyThreads: tmap, treduce, tmapreduce
using Random: AbstractRNG, default_rng
using Statistics: mean

include("abstract.jl")
include("reinforce.jl")
include("reparametrization.jl")
include("pushforward.jl")

export distribution, samples
export REINFORCE

end # module DifferentiableExpectations
