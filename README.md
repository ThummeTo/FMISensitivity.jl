![FMI.jl Logo](https://github.com/ThummeTo/FMI.jl/blob/main/logo/dark/fmijl_logo_640_320.png?raw=true  "FMI.jl Logo")
# FMISensitivity.jl

## What is FMISensitivity.jl?
[*FMISensitivity.jl*](https://github.com/ThummeTo/FMISensitivity.jl) makes FMUs ([fmi-standard.org](http://fmi-standard.org/)) fully differentiable, according to:
- states and derivatives
- inputs, outputs and other observable variables
- parameters

[![Run Tests](https://github.com/ThummeTo/FMISensitivity.jl/actions/workflows/Test.yml/badge.svg)](https://github.com/ThummeTo/FMISensitivity.jl/actions/workflows/Test.yml)
[![Coverage](https://codecov.io/gh/ThummeTo/FMISensitivity.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/ThummeTo/FMISensitivity.jl)
[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac)

## How can I use FMISensitivity.jl?
[*FMISensitivity.jl*](https://github.com/ThummeTo/FMISensitivity.jl) is part of [*FMIFlux.jl*](https://github.com/ThummeTo/FMIFlux.jl). However, if you only need sensitivities without anything around and want to keep the dependencies as small as possible, [*FMISensitivity.jl*](https://github.com/ThummeTo/FMISensitivity.jl) might be the right way to go. You can install it via:

1\. Open a Julia-REPL, switch to package mode using `]`, activate your preferred environment.

2\. Install [*FMISensitivity.jl*](https://github.com/ThummeTo/FMISensitivity.jl):
```julia-repl
(@v1) pkg> add FMISensitivity
```

3\. If you want to check that everything works correctly, you can run the tests bundled with [*FMISensitivity.jl*](https://github.com/ThummeTo/FMISensitivity.jl):
```julia-repl
(@v1) pkg> test FMIImport
```

4\. Have a look inside the [examples folder](https://github.com/ThummeTo/FMI.jl/tree/examples/examples) in the examples branch or the [examples section](https://thummeto.github.io/FMI.jl/dev/examples/overview/) of the documentation of the [*FMI.jl*](https://github.com/ThummeTo/FMI.jl) package. All examples are available as Julia-Script (*.jl*), Jupyter-Notebook (*.ipynb*) and Markdown (*.md*).

## What FMI.jl-Library should I use?
![FMI.jl Family](https://github.com/ThummeTo/FMI.jl/blob/main/docs/src/assets/FMI_JL_family.png?raw=true "FMI.jl Family")
To keep dependencies nice and clean, the original package [*FMI.jl*](https://github.com/ThummeTo/FMI.jl) had been split into new packages:
- [*FMI.jl*](https://github.com/ThummeTo/FMI.jl): High level loading, manipulating, saving or building entire FMUs from scratch
- [*FMIImport.jl*](https://github.com/ThummeTo/FMIImport.jl): Importing FMUs into Julia
- [*FMIExport.jl*](https://github.com/ThummeTo/FMIExport.jl): Exporting stand-alone FMUs from Julia Code
- [*FMICore.jl*](https://github.com/ThummeTo/FMICore.jl): C-code wrapper for the FMI-standard
- [*FMISensitivity.jl*](https://github.com/ThummeTo/FMISensitivity.jl): Sensitivities over FMUs
- [*FMIBuild.jl*](https://github.com/ThummeTo/FMIBuild.jl): Compiler/Compilation dependencies for FMIExport.jl
- [*FMIFlux.jl*](https://github.com/ThummeTo/FMIFlux.jl): Machine Learning with FMUs (differentiation over FMUs)
- [*FMIZoo.jl*](https://github.com/ThummeTo/FMIZoo.jl): A collection of testing and example FMUs

## What Platforms are supported?
[FMISensitivity.jl](https://github.com/ThummeTo/FMISensitivity.jl) is tested (and testing) under Julia Versions *1.6 LTS* and *latest* on Windows *latest* and Ubuntu *latest*. `x64` architectures are tested. Mac and x86-architectures might work, but are not tested.

## How to cite?
Tobias Thummerer, Lars Mikelsons and Josef Kircher. 2021. **NeuralFMU: towards structural integration of FMUs into neural networks.** Martin Sjölund, Lena Buffoni, Adrian Pop and Lennart Ochel (Ed.). Proceedings of 14th Modelica Conference 2021, Linköping, Sweden, September 20-24, 2021. Linköping University Electronic Press, Linköping (Linköping Electronic Conference Proceedings ; 181), 297-306. [DOI: 10.3384/ecp21181297](https://doi.org/10.3384/ecp21181297)

## Related publications?
Tobias Thummerer, Johannes Stoljar and Lars Mikelsons. 2022. **NeuralFMU: presenting a workflow for integrating hybrid NeuralODEs into real-world applications.** Electronics 11, 19, 3202. [DOI: 10.3390/electronics11193202](https://doi.org/10.3390/electronics11193202)

Tobias Thummerer, Johannes Tintenherr, Lars Mikelsons. 2021 **Hybrid modeling of the human cardiovascular system using NeuralFMUs** Journal of Physics: Conference Series 2090, 1, 012155. [DOI: 10.1088/1742-6596/2090/1/012155](https://doi.org/10.1088/1742-6596/2090/1/012155)
