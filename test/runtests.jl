using Aqua
using JET
using JuliaFormatter
using DifferentiableExpectations
using Test

@testset verbose = true "DifferentiableExpectations" begin
    @testset verbose = true "Formalities" begin
        @testset "Aqua" begin
            Aqua.test_all(
                DifferentiableExpectations;
                ambiguities = false,
                deps_compat = (check_extras = false),
            )
        end
        @testset "JET" begin
            JET.test_package(DifferentiableExpectations; target_defined_modules = true)
        end
        @testset "JuliaFormatter" begin
            @test JuliaFormatter.format(
                DifferentiableExpectations;
                verbose = false,
                overwrite = false,
            )
        end
    end
    @testset "REINFORCE" begin
        include("reinforce.jl")
    end
    @testset "Reparametrization" begin
        include("reparametrization.jl")
    end
    @testset "Pushforward" begin
        include("pushforward.jl")
    end
end
