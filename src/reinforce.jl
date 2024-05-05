struct REINFORCE{t,F,D,R<:AbstractRNG} <: AbstractExpectation{t}
    f::F
    rng::R
    nb_samples::Int
end

function REINFORCE(;
    f::F, dist_type::Type{D}, rng::R=default_rng(), nb_samples=1, threaded=true
) where {F,D,R}
    return REINFORCE{threaded,F,D,R}(f, rng, nb_samples)
end

distribution(::REINFORCE{threaded,F,D}, θ...) where {threaded,F,D} = D(θ...)
