module DifferentiableExpectations

using ChainRulesCore: ChainRulesCore, NoTangent, RuleConfig, rrule_via_ad
using Distributions: Distribution
using DocStringExtensions
using OhMyThreads: tmap
using Random: AbstractRNG

include("abstract.jl")
include("reinforce.jl")
include("reparametrization.jl")
include("pushforward.jl")

end # module DifferentiableExpectations
