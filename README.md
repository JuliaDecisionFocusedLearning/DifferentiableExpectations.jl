# DifferentiableExpectations.jl

This package revolves around functions defined as expectations:

$$ F(\theta) = \mathbb{E}_{p_\theta}[f(X)]$$

It allows the computation of derivatives with respect to $\theta$, and their approximation with Monte-Carlo samples.

For more details, refer to the following paper:

> [Monte-Carlo Gradient Estimation in Machine Learning](https://www.jmlr.org/papers/v21/19-346.html), Mohamed et al. (2020)