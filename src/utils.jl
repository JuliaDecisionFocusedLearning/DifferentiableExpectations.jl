"""
    FixKwargs(f, kwargs)

Callable struct that fixes the keyword arguments of `f` to `kwargs...`, and only accepts positional arguments.
"""
struct FixKwargs{F,K}
    f::F
    kwargs::K
end

(fk::FixKwargs)(args...) = fk.f(args...; fk.kwargs...)

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
    mymap(::Val{threaded}, args...)

Apply either `tmap(args...)` or `map(args...)` depending on the value of `threaded`.
"""
mymap(::Val{true}, args...) = tmap(args...)
mymap(::Val{false}, args...) = map(args...)

"""
    mymapreduce(::Val{threaded}, args...)

Apply either `tmapreduce(args...)` or `mapreduce(args...)` depending on the value of `threaded`.
"""
mymapreduce(::Val{true}, args...) = tmapreduce(args...)
mymapreduce(::Val{false}, args...) = mapreduce(args...)

unval(::Val{t}) where {t} = t
