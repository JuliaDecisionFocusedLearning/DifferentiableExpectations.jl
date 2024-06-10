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

maybe_eachcol(x::AbstractVector) = x
maybe_eachcol(x::AbstractMatrix) = eachcol(x)
