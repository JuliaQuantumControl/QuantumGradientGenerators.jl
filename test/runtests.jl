using Test
using SafeTestsets

# Note: comment outer @testset to stop after first @safetestset failure
@time @testset verbose = true "QuantumGradientGenerators" begin

    print("\n* Gradient Generator (test_gradgen.jl):")
    @time @safetestset "Gradient Generator" begin
        include("test_gradgen.jl")
    end

    print("\n* Spectral Radius (test_specrad.jl):")
    @time @safetestset "Spectral Radius" begin
        include("test_specrad.jl")
    end

    print("\n* Interface (test_interface.jl):")
    @time @safetestset "Interface" begin
        include("test_interface.jl")
    end

end;
