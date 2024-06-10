"""
    FixedAtomsProbabilityDistribution

A probability distribution with finite support and fixed atoms.

Whenever its expectation is differentiated, only the weights are considered active, whereas the atoms are considered constant.

# Fields

$(TYPEDFIELDS)
"""
struct FixedAtomsProbabilityDistribution{threaded,A,W<:Real}
    atoms::Vector{A}
    weights::Vector{W}

    function FixedAtomsProbabilityDistribution(
        atoms::Vector{A}, weights::Vector{W}; threaded::Bool=false
    ) where {A,W}
        if isempty(atoms) || isempty(weights)
            throw(ArgumentError("`atoms` and `weights` must be non-empty."))
        elseif length(atoms) != length(weights)
            throw(DimensionMismatch("`atoms` and `weights` must have the same length."))
        elseif !isapprox(sum(weights), one(W); atol=1e-4)
            throw(ArgumentError("`weights` must be normalized to `1`."))
        end
        return new{threaded,A,W}(atoms, weights)
    end
end

Base.length(dist::FixedAtomsProbabilityDistribution) = length(dist.atoms)

function Random.rand(rng::AbstractRNG, dist::FixedAtomsProbabilityDistribution)
    (; atoms, weights) = dist
    return StatsBase.sample(rng, atoms, StatsBase.Weights(weights))
end

function Base.map(f, dist::FixedAtomsProbabilityDistribution{threaded}) where {threaded}
    (; atoms, weights) = dist
    new_atoms = if threaded
        tmap(f, atoms)
    else
        map(f, atoms)
    end
    return FixedAtomsProbabilityDistribution(new_atoms, weights)
end

function Statistics.mean(dist::FixedAtomsProbabilityDistribution{threaded}) where {threaded}
    (; atoms, weights) = dist
    if threaded
        return tmapreduce(*, +, weights, atoms)
    else
        return mapreduce(*, +, weights, atoms)
    end
end

function Statistics.mean(f, dist::FixedAtomsProbabilityDistribution)
    return mean(map(f, dist))
end

function ChainRulesCore.rrule(
    ::typeof(mean), f, dist::FixedAtomsProbabilityDistribution{threaded}
) where {threaded}
    (; atoms, weights) = dist
    new_atoms = if threaded
        tmap(f, atoms)
    else
        map(f, atoms)
    end

    function expectation_pullback(de)
        d_atoms = NoTangent()
        d_weights = if threaded
            tmap(Base.Fix1(dot, de), new_atoms)
        else
            map(Base.Fix1(dot, de), new_atoms)
        end
        d_dist = Tangent{FixedAtomsProbabilityDistribution}(;
            atoms=d_atoms, weights=d_weights
        )
        return NoTangent(), NoTangent(), d_dist
    end

    e = mean(FixedAtomsProbabilityDistribution(new_atoms, weights))
    return e, expectation_pullback
end

function ChainRulesCore.rrule(
    ::typeof(mean), dist::FixedAtomsProbabilityDistribution{threaded}
) where {threaded}
    e, pb = rrule(mean, identity, dist)
    function pb_nof(de)
        p = pb(de)
        return p[1], p[3]
    end
    return e, pb_nof
end
