"""
    FixedAtomsProbabilityDistribution{threaded}

A probability distribution with finite support and fixed atoms.

Whenever its expectation is differentiated, only the weights are considered active, whereas the atoms are considered constant.

# Example

```jldoctest
julia> using DifferentiableExpectations, Statistics, Zygote

julia> dist = FixedAtomsProbabilityDistribution([2, 3], [0.4, 0.6]);

julia> map(abs2, dist)
FixedAtomsProbabilityDistribution{false}([4, 9], [0.4, 0.6])

julia> mean(abs2, dist)
7.0

julia> gradient(mean, abs2, dist)[2]
(atoms = nothing, weights = [4.0, 9.0])
```

# Constructor

    FixedAtomsProbabilityDistribution(
        atoms::Vector,
        weights::Vector;
        threaded=false
    )

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

function Base.show(
    io::IO, dist::FixedAtomsProbabilityDistribution{threaded}
) where {threaded}
    (; atoms, weights) = dist
    return print(io, "FixedAtomsProbabilityDistribution{$threaded}($atoms, $weights)")
end

Base.length(dist::FixedAtomsProbabilityDistribution) = length(dist.atoms)

"""
    rand(rng, dist::FixedAtomsProbabilityDistribution)

Sample from the atoms of `dist` with probability proportional to their weights.
"""
function Random.rand(rng::AbstractRNG, dist::FixedAtomsProbabilityDistribution)
    (; atoms, weights) = dist
    return StatsBase.sample(rng, atoms, StatsBase.Weights(weights))
end

"""
    map(f, dist::FixedAtomsProbabilityDistribution)

Apply `f` to the atoms of `dist`, leave the weights unchanged.
"""
function Base.map(f, dist::FixedAtomsProbabilityDistribution{threaded}) where {threaded}
    (; atoms, weights) = dist
    new_atoms = if threaded
        tmap(f, atoms)
    else
        map(f, atoms)
    end
    return FixedAtomsProbabilityDistribution(new_atoms, weights)
end

"""
    mean(dist::FixedAtomsProbabilityDistribution)

Compute the expectation of `dist`, i.e. the sum of all atoms multiplied by their respective weights.
"""
function Statistics.mean(dist::FixedAtomsProbabilityDistribution{threaded}) where {threaded}
    (; atoms, weights) = dist
    if threaded
        return tmapreduce(*, +, weights, atoms)
    else
        return mapreduce(*, +, weights, atoms)
    end
end

"""
    mean(f, dist::FixedAtomsProbabilityDistribution)

Shortcut for `mean(map(f, dist))`.
"""
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
