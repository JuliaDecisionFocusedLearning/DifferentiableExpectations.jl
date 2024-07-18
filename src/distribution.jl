"""
    FixedAtomsProbabilityDistribution{threaded}

A probability distribution with finite support and fixed atoms.

Whenever its expectation is differentiated, only the weights are considered active, whereas the atoms are considered constant.

# Example

```jldoctest
julia> using DifferentiableExpectations, Statistics, Zygote

julia> using DifferentiableExpectations: atoms, weights

julia> dist = FixedAtomsProbabilityDistribution([2, 3], [0.4, 0.6]);

julia> atoms(map(abs2, dist))
2-element Vector{Int64}:
 4
 9

julia> weights(map(abs2, dist))
2-element Vector{Float64}:
 0.4
 0.6

julia> mean(abs2, dist)
7.0

julia> gradient(mean, abs2, dist)[2]
(atoms = nothing, weights = [4.0, 9.0])
```

# Constructor

    FixedAtomsProbabilityDistribution(
        atoms::AbstractVector,
        weights::AbstractVector=uniform_weights(atoms);
        threaded=false
    )

# Fields

$(TYPEDFIELDS)
"""
struct FixedAtomsProbabilityDistribution{
    threaded,A<:AbstractVector,W<:AbstractVector{<:Real}
}
    atoms::A
    weights::W

    function FixedAtomsProbabilityDistribution(
        atoms::A, weights::W=uniform_weights(atoms); threaded::Bool=false
    ) where {A,W}
        if isempty(atoms) || isempty(weights)
            throw(ArgumentError("`atoms` and `weights` must be non-empty."))
        elseif length(atoms) != length(weights)
            throw(DimensionMismatch("`atoms` and `weights` must have the same length."))
        elseif !isapprox(sum(weights), one(eltype(weights)); atol=1e-4)
            throw(ArgumentError("`weights` must be normalized to `1`."))
        end
        return new{threaded,A,W}(atoms, weights)
    end
end

"""
    atoms(dist::FixedAtomsProbabilityDistribution)

Get the vector of atoms of a distribution.
"""
atoms(dist::FixedAtomsProbabilityDistribution) = dist.atoms

"""
    weights(dist::FixedAtomsProbabilityDistribution)

Get the vector of weights of a distribution.
"""
weights(dist::FixedAtomsProbabilityDistribution) = dist.weights

is_threaded(::FixedAtomsProbabilityDistribution{t}) where {t} = Val(t)

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
function Base.map(f, dist::FixedAtomsProbabilityDistribution)
    (; atoms, weights) = dist
    new_atoms = mymap(is_threaded(dist), f, atoms)
    return FixedAtomsProbabilityDistribution(new_atoms, weights)
end

"""
    mean(dist::FixedAtomsProbabilityDistribution)

Compute the expectation of `dist`, i.e. the sum of all atoms multiplied by their respective weights.
"""
function Statistics.mean(dist::FixedAtomsProbabilityDistribution)
    (; atoms, weights) = dist
    return mymapreduce(is_threaded(dist), *, +, weights, atoms)
end

"""
    mean(f, dist::FixedAtomsProbabilityDistribution)

Shortcut for `mean(map(f, dist))`.
"""
function Statistics.mean(f, dist::FixedAtomsProbabilityDistribution)
    return mean(map(f, dist))
end

function ChainRulesCore.rrule(::typeof(mean), dist::FixedAtomsProbabilityDistribution)
    (; atoms) = dist
    e = mean(dist)
    function dist_mean_pullback(Δe)
        Δatoms = NoTangent()
        Δweights = mymap(is_threaded(dist), Base.Fix1(dot, Δe), atoms)
        Δdist = Tangent{FixedAtomsProbabilityDistribution}(; atoms=Δatoms, weights=Δweights)
        return NoTangent(), Δdist
    end
    return e, dist_mean_pullback
end

function ChainRulesCore.rrule(::typeof(mean), f, dist::FixedAtomsProbabilityDistribution)
    new_dist = map(f, dist)
    new_atoms = new_dist.atoms
    e = mean(new_dist)
    function dist_fmean_pullback(Δe)
        Δatoms = NoTangent()
        Δweights = mymap(is_threaded(dist), Base.Fix1(dot, Δe), new_atoms)
        Δdist = Tangent{FixedAtomsProbabilityDistribution}(; atoms=Δatoms, weights=Δweights)
        return NoTangent(), NoTangent(), Δdist
    end
    return e, dist_fmean_pullback
end
