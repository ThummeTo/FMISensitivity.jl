![FMI.jl Logo](https://github.com/ThummeTo/FMI.jl/blob/main/logo/dark/fmijl_logo_640_320.png?raw=true  "FMI.jl Logo")
# FMISensitivity.jl

## What is FMISensitivity.jl?
Unfortunately, FMUs ([fmi-standard.org](http://fmi-standard.org/)) are not differentiable by design. 
To enable their full potential inside Julia, [*FMISensitivity.jl*](https://github.com/ThummeTo/FMISensitivity.jl) makes FMUs fully differentiable, regarding to:
- states and derivatives
- inputs, outputs and other observable variables
- parameters
- event indicators 
- explicit time
- state change sensitivity by event $\partial x^{+} / \partial x^{-}$ (if paired with *FMIFlux.jl*)

This opens up to many applications like:
- FMUs in Scientific Machine Learning, for example as part of Neural(O)DEs or PINNs with [*FMIFlux.jl*](https://github.com/ThummeTo/FMIFlux.jl)
- gradient-based optimization of FMUs (typically parameters) with [*FMI.jl*](https://github.com/ThummeTo/FMIFlux.jl) (also *dynamic* optimization)
- linearization, linear analysis and controller design
- adding directional derivatives for existing FMUs with the power of Julia AD and [*FMIExport.jl*](https://github.com/ThummeTo/FMIExport.jl)
- ...

Supported AD-Frameworks are:
- ForwardDiff
- FiniteDiff
- ReverseDiff
- Zygote [WIP]

Here, *FMISensitivity.jl* uses everything the FMI-standard and Julia currently offers:
- FMI built-in directional derivatives and adjoint derivatives [WIP]
- Finite Differences (by *FiniteDiff.jl*) for FMUs that don't offer sensitivity information, as well as for special derivatives that are not part of the FMI-standard (like e.g. event-indicators or explicit time)
- coloring based on sparsity information shipped with the FMU [WIP]
- coloring based on sparsity detection for FMUs without sparsity information [WIP]
- implicite differentation
- ...

[![Run Tests](https://github.com/ThummeTo/FMISensitivity.jl/actions/workflows/Test.yml/badge.svg)](https://github.com/ThummeTo/FMISensitivity.jl/actions/workflows/Test.yml)
[![Coverage](https://codecov.io/gh/ThummeTo/FMISensitivity.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/ThummeTo/FMISensitivity.jl)
[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac)

## How can I use FMISensitivity.jl?
[*FMISensitivity.jl*](https://github.com/ThummeTo/FMISensitivity.jl) is part of [*FMIFlux.jl*](https://github.com/ThummeTo/FMIFlux.jl). If you only need FMU sensitivities without anything around and want to keep the dependencies as small as possible, [*FMISensitivity.jl*](https://github.com/ThummeTo/FMISensitivity.jl) might be the right way to go. You can install it via:

1\. Open a Julia-REPL, switch to package mode using `]`, activate your preferred environment.

2\. Install [*FMISensitivity.jl*](https://github.com/ThummeTo/FMISensitivity.jl):
```julia-repl
(@v1) pkg> add FMISensitivity
```

3\. If you want to check that everything works correctly, you can run the tests bundled with [*FMISensitivity.jl*](https://github.com/ThummeTo/FMISensitivity.jl):
```julia-repl
(@v1) pkg> test FMISensitivity
```

4\. Have a look inside the [examples folder](https://github.com/ThummeTo/FMI.jl/tree/examples/examples) in the examples branch or the [examples section](https://thummeto.github.io/FMI.jl/dev/examples/overview/) of the documentation of the [*FMI.jl*](https://github.com/ThummeTo/FMI.jl) package. All examples are available as Julia-Script (*.jl*), Jupyter-Notebook (*.ipynb*) and Markdown (*.md*).

## What FMI.jl-Library should I use?
![FMI.jl Family](https://github.com/ThummeTo/FMI.jl/blob/main/docs/src/assets/FMI_JL_family.png?raw=true "FMI.jl Family")
To keep dependencies nice and clean, the original package [*FMI.jl*](https://github.com/ThummeTo/FMI.jl) had been split into new packages:
- [*FMI.jl*](https://github.com/ThummeTo/FMI.jl): High level loading, manipulating, saving or building entire FMUs from scratch
- [*FMIImport.jl*](https://github.com/ThummeTo/FMIImport.jl): Importing FMUs into Julia
- [*FMIExport.jl*](https://github.com/ThummeTo/FMIExport.jl): Exporting stand-alone FMUs from Julia Code
- [*FMICore.jl*](https://github.com/ThummeTo/FMICore.jl): C-code wrapper for the FMI-standard
- [*FMISensitivity.jl*](https://github.com/ThummeTo/FMISensitivity.jl): Static and dynamic sensitivities over FMUs
- [*FMIBuild.jl*](https://github.com/ThummeTo/FMIBuild.jl): Compiler/Compilation dependencies for FMIExport.jl
- [*FMIFlux.jl*](https://github.com/ThummeTo/FMIFlux.jl): Machine Learning with FMUs (differentiation over FMUs)
- [*FMIZoo.jl*](https://github.com/ThummeTo/FMIZoo.jl): A collection of testing and example FMUs

## What Platforms are supported?
[FMISensitivity.jl](https://github.com/ThummeTo/FMISensitivity.jl) is tested (and testing) under Julia Versions *1.6 LTS* and *latest* on Windows *latest* and Ubuntu *latest*. `x64` architectures are tested. Mac and x86-architectures might work, but are not tested.

## How to cite?
Coming soon ...

Tobias Thummerer, Lars Mikelsons and Josef Kircher. 2021. **NeuralFMU: towards structural integration of FMUs into neural networks.** Martin Sjölund, Lena Buffoni, Adrian Pop and Lennart Ochel (Ed.). Proceedings of 14th Modelica Conference 2021, Linköping, Sweden, September 20-24, 2021. Linköping University Electronic Press, Linköping (Linköping Electronic Conference Proceedings ; 181), 297-306. [DOI: 10.3384/ecp21181297](https://doi.org/10.3384/ecp21181297)