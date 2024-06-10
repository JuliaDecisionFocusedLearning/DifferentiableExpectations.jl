# DifferentiableExpectations.jl

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JuliaDecisionFocusedLearning.github.io/DifferentiableExpectations.jl/dev/)
[![Build Status](https://github.com/JuliaDecisionFocusedLearning/DifferentiableExpectations.jl/actions/workflows/Test.yml/badge.svg?branch=main)](https://github.com/JuliaDecisionFocusedLearning/DifferentiableExpectations.jl/actions/workflows/Test.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/JuliaDecisionFocusedLearning/DifferentiableExpectations.jl/branch/main/graph/badge.svg)](https://app.codecov.io/gh/JuliaDecisionFocusedLearning/DifferentiableExpectations.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/JuliaDiff/BlueStyle)

A Julia package for differentiating through expectations with Monte-Carlo estimates.

It allows the computation of approximate derivatives for functions of the form

```math
F(\theta) = \mathbb{E}_{p(\theta)}[f(X)]
```

The following estimators are implemented:

  - [REINFORCE](https://jmlr.org/papers/volume21/19-346/19-346.pdf#section.20)
  - [Reparametrization](https://jmlr.org/papers/volume21/19-346/19-346.pdf#section.56)
