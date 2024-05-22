#
# Copyright (c) 2023 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

using FMIImport
using Test
import Random
using FMIZoo

using FMIImport.FMIBase.FMICore: fmi2Integer, fmi2Boolean, fmi2Real, fmi2String
using FMIImport.FMIBase: FMU_EXECUTION_CONFIGURATIONS

exportingToolsWindows = [("Dymola", "2022x")]
exportingToolsLinux = [("Dymola", "2022x")]

global EXPORTINGTOOL
global EXPORTINGVERSION

function runtestsFMI2(exportingTool)
    global EXPORTINGTOOL, EXPORTINGVERSION
    EXPORTINGTOOL = exportingTool[1]
    EXPORTINGVERSION = exportingTool[2]

    @testset "Testing FMUs exported from $exportingTool" begin
        @testset "Jacobians / Gradients" begin
            include("FMI2/jacobians_gradients.jl")
        end

        @testset "Solution" begin
            include("FMI2/solution.jl")
        end
    end
end

# enable assertions for warnings/errors for all default execution configurations 
for exec in FMU_EXECUTION_CONFIGURATIONS
    exec.assertOnError = true
    exec.assertOnWarning = true
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
