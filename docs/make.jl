using DifferentiableExpectations
using Documenter

cp(joinpath(@__DIR__, "..", "README.md"), joinpath(@__DIR__, "src", "index.md"); force=true)

makedocs(;
    modules=[DifferentiableExpectations],
    authors="Members of JuliaDecisionFocusedLearning",
    sitename="DifferentiableExpectations.jl",
    format=Documenter.HTML(),
    pages=["Home" => "index.md", "API reference" => "api.md"],
)

deploydocs(;
    repo="github.com/JuliaDecisionFocusedLearning/DifferentiableExpectations.jl",
    devbranch="main",
)
