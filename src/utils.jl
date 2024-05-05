function tmapmean(f, args...)
    return tmapreduce(f, +, args...) / length(first(args))
end

function tmean(args...)
    return treduce(+, args...) / length(first(args))
end
