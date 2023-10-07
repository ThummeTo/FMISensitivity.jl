#
# Copyright (c) 2023 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

using FMIImport
using Test
import Random
using FMIZoo

using FMIImport.FMICore: fmi2Integer, fmi2Boolean, fmi2Real, fmi2String
using FMIImport.FMICore: FMU2_EXECUTION_CONFIGURATIONS

exportingToolsWindows = [("Dymola", "2022x")]
exportingToolsLinux = [("Dymola", "2022x")]

function runtestsFMI2(exportingTool)
    ENV["EXPORTINGTOOL"] = exportingTool[1]
    ENV["EXPORTINGVERSION"] = exportingTool[2]

    # enable assertions for warnings/errors for all default execution configurations 
    for exec in [FMU2_EXECUTION_CONFIGURATION_NO_FREEING, FMU2_EXECUTION_CONFIGURATION_NO_RESET, FMU2_EXECUTION_CONFIGURATION_RESET, FMU2_EXECUTION_CONFIGURATION_NOTHING]
        exec.assertOnError = true
        exec.assertOnWarning = true
    end

    @testset "Testing FMUs exported from $exportingTool" begin
        @testset "Sensitivities" begin
            include("FMI2.jl")
        end
    end
end

@testset "FMIImport.jl" begin
    if Sys.iswindows()
        @info "Automated testing is supported on Windows."
        for exportingTool in exportingToolsWindows
            runtestsFMI2(exportingTool)
        end
    elseif Sys.islinux()
        @info "Automated testing is supported on Linux."
        for exportingTool in exportingToolsLinux
            runtestsFMI2(exportingTool)
        end
    elseif Sys.isapple()
        @warn "Test-sets are currrently using Windows- and Linux-FMUs, automated testing for macOS is currently not supported."
    end
end
