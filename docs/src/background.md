# Background

Consider a function ``f: \mathbb{R}^n \to \mathbb{R}^m`` and a parametric probability distribution ``p(\theta)`` on the input space ``\mathbb{R}^n``.
Given a random variable ``X \sim p(\theta)``, we want to differentiate the following expectation with respect to ``\theta``:

```math
F(\theta) = \mathbb{E}_{p(\theta)}[f(X)]
```

Since ``F`` is a vector-to-vector function, the key quantity we want to compute is its Jacobian matrix ``\partial F(\theta) \in \mathbb{R}^{m \times n}``.
However, to implement automatic differentiation, we only need vector-Jacobian products (VJPs) ``\partial F(\theta)^\top v`` with ``v \in \mathbb{R}^m``, also called pullbacks.
See the book by [blondelElementsDifferentiableProgramming2024](@citet) to know more.

Most of the math below is taken from [mohamedMonteCarloGradient2020](@citet).

## Empirical distribution

Implemented by [`FixedAtomsProbabilityDistribution`](@ref).

### Mean

An empirical distribution ``\pi`` is given by a set of atoms ``a_i`` and associated weights ``w_i \geq 0`` such that ``\sum_i w_i = 1``.
Its expectation is ``\mathbb{E}[\pi] = \sum w_i a_i``.
The gradient of this expectation with respect to weight ``w_j`` is ``\partial_{w_j} \mathbb{E}[\pi] = a_j``.
Thus, the vector-Jacobian product is:

```math
\left(\partial_{w_j} \mathbb{E}[\pi]\right)^\top v = a_j^\top v 
```

We assume the atoms are kept constant during differentiation.

## REINFORCE

Implemented by [`Reinforce`](@ref).

### Principle

The REINFORCE estimator is derived with the help of the identity ``\nabla \log u = \nabla u / u``:

```math
\begin{aligned}
F(\theta + \varepsilon)
& = \int f(x) ~ p(x, \theta + \varepsilon) ~ \mathrm{d}x \\
& \approx \int f(x) ~ \left(p(x, \theta) + \nabla_\theta p(x, \theta)^\top \varepsilon\right) ~ \mathrm{d}x \\
& = \int f(x) ~ \left(p(x, \theta) + p(x, \theta) \nabla_\theta \log p(x, \theta)^\top \varepsilon\right) ~ \mathrm{d}x \\
& = F(\theta) + \left(\int f(x) ~ p(x, \theta) \nabla_\theta \log p(x, \theta)^\top ~ \mathrm{d}x\right) \varepsilon \\
& = F(\theta) + \mathbb{E}_{p(\theta)} \left[f(X) \nabla_\theta \log p(X, \theta)^\top\right] ~ \varepsilon \\
\end{aligned}
```

We thus identify the Jacobian matrix:

```math
\partial F(\theta) = \mathbb{E}_{p(\theta)} \left[f(X) \nabla_\theta \log p(X, \theta)^\top\right]
```

And the vector-Jacobian product:

```math
\partial F(\theta)^\top v = \mathbb{E}_{p(\theta)} \left[(f(X)^\top v) \nabla_\theta \log p(X, \theta)\right]
```

### Variance reduction

Since the REINFORCE estimator has high variance, it can be reduced by using a baseline [koolBuyREINFORCESamples2022](@citep).
For $k > 1$ Monte-Carlo samples, we have

```math
\begin{aligned}
\partial F(\theta) &\simeq \frac{1}{k}\sum_{i=1}^k f(x_k) \nabla_\theta\log p(x_k, \theta)^\top\\
& \simeq \frac{1}{k}\sum_{i=1}^k \left(f(x_i) - \frac{1}{k - 1}\sum_{j\neq i} f(x_j)\right) \nabla_\theta\log p(x_i, \theta)^\top\\
& = \frac{1}{k - 1}\sum_{i=1}^k \left(f(x_i) - \frac{1}{k}\sum_{j=1}^k f(x_j)\right) \nabla_\theta\log p(x_i, \theta)^\top
\end{aligned}
```

This gives the following vector-Jacobian product:

```math
\partial F(\theta)^\top v \simeq \frac{1}{k - 1}\sum_{i=1}^k \left(\left(f(x_i) - \frac{1}{k}\sum_{j=1}^k f(x_j)\right)^\top v\right) \nabla_\theta\log p(x_i, \theta)
```

## Reparametrization

Implemented by [`Reparametrization`](@ref).

### Trick

The reparametrization trick assumes that we can rewrite the random variable ``X \sim p(\theta)`` as ``X = g(Z, \theta)``, where ``Z \sim q`` is another random variable whose distribution does not depend on ``\theta``.

```math
\begin{aligned}
F(\theta + \varepsilon)
& = \int f(g(z, \theta + \varepsilon)) ~ q(z) ~ \mathrm{d}z \\
& \approx \int f\left(g(z, \theta) + \partial_\theta g(z, \theta) ~ \varepsilon\right) ~ q(z) ~ \mathrm{d}z \\
& \approx F(\theta) + \int \partial f(g(z, \theta)) ~ \partial_\theta g(z, \theta) ~ \varepsilon ~ q(z) ~ \mathrm{d}z \\
& \approx F(\theta) + \mathbb{E}_q \left[ \partial f(g(Z, \theta)) ~ \partial_\theta g(Z, \theta) \right] ~ \varepsilon \\
\end{aligned}
```

If we denote ``h(z, \theta) = f(g(z, \theta))``, then we identify the Jacobian matrix:

```math
\partial F(\theta) = \mathbb{E}_q \left[ \partial_\theta h(Z, \theta) \right]
```

And the vector-Jacobian product:

```math
\partial F(\theta)^\top v = \mathbb{E}_q \left[ \partial_\theta h(Z, \theta)^\top v \right]
```

### Catalogue

The following reparametrizations are implemented:

- Univariate Normal: ``X \sim \mathcal{N}(\mu, \sigma^2)`` is equivalent to ``X = \mu + \sigma Z`` with ``Z \sim \mathcal{N}(0, 1)``.
- Multivariate Normal: ``X \sim \mathcal{N}(\mu, \Sigma)`` is equivalent to ``X = \mu + L Z`` with ``Z \sim \mathcal{N}(0, I)`` and ``L L^\top = \Sigma``. The matrix ``L`` can be obtained by Cholesky decomposition of ``\Sigma``.

## Bibliography

```@bibliography
```
