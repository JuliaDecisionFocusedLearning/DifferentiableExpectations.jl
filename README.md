# DifferentiableExpectations.jl

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JuliaDecisionFocusedLearning.github.io/DifferentiableExpectations.jl/dev/)
[![Build Status](https://github.com/JuliaDecisionFocusedLearning/DifferentiableExpectations.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JuliaDecisionFocusedLearning/DifferentiableExpectations.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/JuliaDecisionFocusedLearning/DifferentiableExpectations.jl/branch/main/graph/badge.svg)](https://app.codecov.io/gh/JuliaDecisionFocusedLearning/DifferentiableExpectations.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/JuliaDiff/BlueStyle)

This package revolves around functions defined as expectations:

```math
F(\theta) = \mathbb{E}_{p_\theta}[f(X)]
```

It allows the computation of derivatives with respect to $\theta$, and their approximation with Monte-Carlo samples.

For more details, refer to the following paper:

> [Monte-Carlo Gradient Estimation in Machine Learning](https://www.jmlr.org/papers/v21/19-346.html), Mohamed et al. (2020)