#
# Copyright (c) 2023 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

using FMIImport
using Test
import Random
using FMIZoo

using FMISensitivity
using FMISensitivity.FMIBase
using FMISensitivity.FMIBase: FMU_EXECUTION_CONFIGURATIONS

fmuStructs = ("FMU", "FMUCOMPONENT")

# enable assertions for warnings/errors for all default execution configurations 
for exec in FMU_EXECUTION_CONFIGURATIONS
    exec.assertOnError = true
    exec.assertOnWarning = true
end

function getFMUStruct(modelname, mode, tool=ENV["EXPORTINGTOOL"], version=ENV["EXPORTINGVERSION"], fmiversion=ENV["FMIVERSION"], fmustruct=ENV["FMUSTRUCT"]; kwargs...)
    
    # choose FMU or FMUComponent
    if endswith(modelname, ".fmu")
        fmu = loadFMU(modelname; kwargs...)
    else
        fmu = loadFMU(modelname, tool, version, fmiversion; kwargs...) 
    end

    if fmustruct == "FMU"
        return fmu, fmu

    elseif fmustruct == "FMUCOMPONENT"
        inst, _ = FMIImport.prepareSolveFMU(fmu, nothing, mode; loggingOn=true)
        @test !isnothing(inst)
        return inst, fmu

    else
        @assert false "Unknown fmuStruct, variable `FMUSTRUCT` = `$(fmustruct)`"
    end
end

@testset "FMIImport.jl" begin
    if Sys.iswindows() || Sys.islinux()
        @info "Automated testing is supported on Windows/Linux."
        
        ENV["EXPORTINGTOOL"] = "Dymola"
        ENV["EXPORTINGVERSION"] = "2023x"

        for fmiversion in (2.0, 3.0)
            ENV["FMIVERSION"] = fmiversion

            @testset "Testing FMI $(ENV["FMIVERSION"]) FMUs exported from $(ENV["EXPORTINGTOOL"]) $(ENV["EXPORTINGVERSION"])" begin

                ENV["FMUSTRUCT"] = "FMUCOMPONENT"

                @testset "Functions for $(ENV["FMUSTRUCT"])" begin
                    @testset "Jacobians / Gradients" begin
                        include("jacobians_gradients.jl")
                    end
            
                    @testset "Solution" begin
                        include("solution.jl")
                    end
                end

                # this script requires additional libraries
                # @testset "Performance" begin
                #     include("performance.jl")
                # end
            end
        end
    
    elseif Sys.isapple()
        @warn "Test-sets are currrently using Windows- and Linux-FMUs, automated testing for macOS is currently not supported."
    end
end
