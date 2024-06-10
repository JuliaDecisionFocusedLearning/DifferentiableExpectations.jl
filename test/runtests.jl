using Aqua: Aqua
using Documenter: Documenter
using JET: JET
using JuliaFormatter: JuliaFormatter
using DifferentiableExpectations
using Test
using Zygote

@testset verbose = true "DifferentiableExpectations" begin
    @testset verbose = true "Formalities" begin
        @testset "Aqua" begin
            Aqua.test_all(
                DifferentiableExpectations;
                ambiguities=false,
                deps_compat=(check_extras = false),
            )
        end
        @testset "JET" begin
            JET.test_package(DifferentiableExpectations; target_defined_modules=true)
        end
        @testset "JuliaFormatter" begin
            @test JuliaFormatter.format(
                DifferentiableExpectations; verbose=false, overwrite=false
            )
        end
        @testset "Documenter" begin
            Documenter.doctest(DifferentiableExpectations)
        end
    end
    @testset verbose = true "Expectation" begin
        include("expectation.jl")
    end
    @testset "Distribution" begin
        include("distribution.jl")
    end
end
