"""
    FixKwargs(f, kwargs)

Callable struct that fixes the keyword arguments of `f` to `kwargs...`, and only accepts positional arguments.
"""
struct FixKwargs{F,K}
    f::F
    kwargs::K
end

function (fk::FixKwargs)(args...)
    return fk.f(args...; fk.kwargs...)
end

function tmean(args...)
    return treduce(+, args...) / length(first(args))
end

"""
    maybe_eachcol(x::AbstractVector)

Return `x`.
"""
maybe_eachcol(x::AbstractVector) = x

"""
    maybe_eachcol(x::AbstractMatrix)

Return `eachcol(x)`.
"""
maybe_eachcol(x::AbstractMatrix) = eachcol(x)

uniform_weights(x::AbstractArray) = ones(size(x)) ./ prod(size(x))

"""
    tmap_or_map(::SomeType{threaded}, args...)

Apply either `tmap(args...)` or `map(args...)` depending on the value of `threaded`.
"""
function tmap_or_map end

"""
    tmapreduce_or_mapreduce(::SomeType{threaded}, args...)

Apply either `tmapreduce(args...)` or `mapreduce(args...)` depending on the value of `threaded`.
"""
function tmapreduce_or_mapreduce end

"""
    tmean_or_mean(::SomeType{threaded}, args...)

Apply either `tmean(args...)` or `mean(args...)` depending on the value of `threaded`.
"""
function tmean_or_mean end
