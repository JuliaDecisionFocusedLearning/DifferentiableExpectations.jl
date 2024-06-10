using DifferentiableExpectations
using Documenter
using DocumenterCitations

bib = CitationBibliography(joinpath(@__DIR__, "src", "DiffExp.bib"); style=:authoryear)

cp(joinpath(@__DIR__, "..", "README.md"), joinpath(@__DIR__, "src", "index.md"); force=true)

makedocs(;
    modules=[DifferentiableExpectations],
    authors="Members of JuliaDecisionFocusedLearning",
    sitename="DifferentiableExpectations.jl",
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",  #
        "API reference" => "api.md",
        "Background" => "background.md",
    ],
    plugins=[bib],
)

deploydocs(;
    repo="github.com/JuliaDecisionFocusedLearning/DifferentiableExpectations.jl",
    devbranch="main",
)
