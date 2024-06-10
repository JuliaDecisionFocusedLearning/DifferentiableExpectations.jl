struct FixKwargs{F,K}
    f::F
    kwargs::K
end

function (fk::FixKwargs)(args...)
    return fk.f(args...; fk.kwargs...)
end

function tmapmean(f, args...)
    return tmapreduce(f, +, args...) / length(first(args))
end

function tmean(args...)
    return treduce(+, args...) / length(first(args))
end
