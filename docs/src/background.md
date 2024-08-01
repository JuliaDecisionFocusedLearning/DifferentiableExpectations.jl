# Background

Most of the math below is taken from [mohamedMonteCarloGradient2020](@citet).

Consider a function $f: \mathbb{R}^n \to \mathbb{R}^m$, a parameter $\theta \in \mathbb{R}^d$ and a parametric probability distribution $p(\theta)$ on the input space.
Given a random variable $X \sim p(\theta)$, we want to differentiate the expectation of $Y = f(X)$ with respect to $\theta$:

$$E(\theta) = \mathbb{E}[f(X)] = \int f(x) ~ p(x | \theta) ~\mathrm{d} x$$

Usually this is approximated with Monte-Carlo sampling: let $x_1, \dots, x_S \sim p(\theta)$ be i.i.d., we have the estimator

$$E(\theta) \simeq \frac{1}{S} \sum_{s=1}^S f(x_s)$$

## Autodiff

Since $E$ is a vector-to-vector function, the key quantity we want to compute is its Jacobian matrix $\partial E(\theta) \in \mathbb{R}^{m \times n}$:

$$\partial E(\theta) = \int y ~ \nabla_\theta q(y | \theta)^\top ~ \mathrm{d} y = \int f(x) ~ \nabla_\theta p(x | \theta)^\top ~\mathrm{d} x$$

However, to implement automatic differentiation, we only need the vector-Jacobian product (VJP) $\partial E(\theta)^\top \bar{y}$ with an output cotangent $\bar{y} \in \mathbb{R}^m$.
See the book by [blondelElementsDifferentiableProgramming2024](@citet) to know more.

Our goal is to rephrase this VJP as an expectation, so that we may approximate it with Monte-Carlo sampling as well.

## REINFORCE

Implemented by [`Reinforce`](@ref).

### Score function

The REINFORCE estimator is derived with the help of the identity $\nabla \log u = \nabla u / u$:

$$\begin{aligned}
\partial E(\theta)
& = \int f(x) ~ \nabla_\theta p(x | \theta)^\top ~ \mathrm{d}x \\
& = \int f(x) ~ \nabla_\theta \log p(x | \theta)^\top p(x | \theta) ~ \mathrm{d}x \\
& = \mathbb{E} \left[f(X) \nabla_\theta \log p(X | \theta)^\top\right] \\
\end{aligned}$$

And the VJP:

$$\partial E(\theta)^\top \bar{y} = \mathbb{E} \left[f(X)^\top \bar{y} ~\nabla_\theta \log p(X | \theta)\right]$$

Our Monte-Carlo approximation will therefore be:

$$\partial E(\theta)^\top \bar{y} \simeq \frac{1}{S} \sum_{s=1}^S f(x_s)^\top \bar{y} ~ \nabla_\theta \log p(x_s | \theta)$$

### Variance reduction

The REINFORCE estimator has high variance, but its variance is reduced by subtracting a so-called baseline $b = \frac{1}{S} \sum_{s=1}^S f(x_s)$ [koolBuyREINFORCESamples2022](@citep).

For $S > 1$ Monte-Carlo samples, we have

$$\begin{aligned}
\partial E(\theta)^\top \bar{y} 
& \simeq \frac{1}{S} \sum_{s=1}^S \left(f(x_s) - \frac{1}{S - 1}\sum_{j\neq s} f(x_j) \right)^\top \bar{y} ~ \nabla_\theta\log p(x_s | \theta)\\
& = \frac{1}{S - 1}\sum_{s=1}^S (f(x_s) - b)^\top \bar{y} ~ \nabla_\theta\log p(x_s | \theta)
\end{aligned}$$

## Reparametrization

Implemented by [`Reparametrization`](@ref).

### Trick

The reparametrization trick assumes that we can rewrite the random variable $X \sim p(\theta)$ as $X = g_\theta(Z)$, where $Z \sim r$ is another random variable whose distribution $r$ does not depend on $\theta$.

The expectation is rewritten with $h = f \circ g$:

$$E(\theta) = \mathbb{E}\left[ f(g_\theta(Z)) \right] = \mathbb{E}\left[ h_\theta(Z) \right]$$

And we can directly differentiate through the expectation:

$$\partial E(\theta) = \mathbb{E} \left[ \partial_\theta h_\theta(Z) \right]$$

This yields the VJP:

$$\partial E(\theta)^\top \bar{y} = \mathbb{E} \left[ \partial_\theta h_\theta(Z)^\top \bar{y} \right]$$

We can use a Monte-Carlo approximation with i.i.d. samples $z_1, \dots, z_S \sim r$:

$$\partial E(\theta)^\top \bar{y} \simeq \frac{1}{S} \sum_{s=1}^S \partial_\theta h_\theta(z_s)^\top \bar{y}$$

### Catalogue

The following reparametrizations are implemented:

- Univariate Normal: $X \sim \mathcal{N}(\mu, \sigma^2)$ is equivalent to $X = \mu + \sigma Z$ with $Z \sim \mathcal{N}(0, 1)$.
- Multivariate Normal: $X \sim \mathcal{N}(\mu, \Sigma)$ is equivalent to $X = \mu + L Z$ with $Z \sim \mathcal{N}(0, I)$ and $L L^\top = \Sigma$. The matrix $L$ can be obtained by Cholesky decomposition of $\Sigma$.

## Probability gradients

In the case where $f$ is a function that takes values in a finite set $\mathcal{Y} = \{y_1, \cdots, y_K\}$, we may also want to compute the jacobian of the probability weights vector:

$$q : \theta \longmapsto \begin{pmatrix} q(y_1|\theta) = \mathbb{P}(f(X) = y_1|\theta) \\ \dots \\ q(y_K|\theta) = \mathbb{P}(f(X) = y_K|\theta) \end{pmatrix}$$

whose Jacobian is given by

$$\partial_\theta q(\theta) = \begin{pmatrix} \nabla_\theta q(y_1|\theta)^\top \\ \dots \\ \nabla_\theta q(y_K|\theta)^\top \end{pmatrix}$$

### REINFORCE probability gradients

The REINFORCE technique can be applied in a similar way:

$$q(y_k | \theta) = \mathbb{E}[\mathbf{1}\{f(X) = y_k\}]  = \int \mathbf{1} \{f(x) = y_k\} ~ p(x | \theta) ~ \mathrm{d}x$$

Differentiating through the integral,

$$\begin{aligned}
\nabla_\theta q(y_k | \theta)
& = \int \mathbf{1} \{f(x) = y_k\} ~ \nabla_\theta p(x | \theta) ~ \mathrm{d}x \\
& = \mathbb{E} [\mathbf{1} \{f(X) = y_k\} ~ \nabla_\theta \log p(X | \theta)]
\end{aligned}$$

The Monte-Carlo approximation for this is

$$\nabla_\theta q(y_k | \theta) \simeq \frac{1}{S} \sum_{s=1}^S \mathbf{1} \{f(x_s) = y_k\} ~ \nabla_\theta \log p(x_s | \theta)$$

The VJP is then

$$\begin{aligned}
\partial_\theta q(\theta)^\top \bar{q} &= \sum_{k=1}^K \bar{q}_k \nabla_\theta q(y_k | \theta)\\
&\simeq  \frac{1}{S} \sum_{s=1}^S \left[\sum_{k=1}^K \bar{q}_k \mathbf{1} \{f(x_s) = y_k\}\right] ~ \nabla_\theta \log p(x_s | \theta)
\end{aligned}$$

In our implementation, the [`empirical_distribution`](@ref) method outputs an empirical [`FixedAtomsProbabilityDistribution`](@ref) whiich uniform weights $\frac{1}{S}$, where some $x_s$ can be repeated.

$$q : \theta \longmapsto \begin{pmatrix} q(f(x_1)|\theta) \\ \dots \\ q(f(x_S) | \theta) \end{pmatrix}$$

We therefore define the corresponding VJP as

$$\partial_\theta q(\theta)^\top \bar{q} = \frac{1}{S} \sum_s \bar{q}_s \nabla_\theta \log p(x_s | \theta)$$

If $\bar q$ comes from `mean`, we have $\bar q_s = f(x_s)^\top \bar y$ and we obtain the REINFORCE VJP.

This VJP can be interpreted as an empirical expectation, to which we can also apply variance reduction:
$$\partial_\theta q(\theta)^\top \bar q \approx \frac{1}{S-1}\sum_s(\bar q_s - b') \nabla_\theta \log p(x_s|\theta)$$
with $b' = \frac{1}{S}\sum_s \bar q_s$.

Again, if $\bar q$ comes from `mean`, we have $\bar q_s = f(x_s)^\top \bar y$ and $b' = b^\top \bar y$. We then obtain the REINFORCE backward rule with variance reduction:
$$\partial_\theta q(\theta)^\top \bar q \approx \frac{1}{S-1}\sum_s(f(x_s) - b)^\top \bar y \nabla_\theta \log p(x_s|\theta)$$

### Reparametrization probability gradients

To leverage reparametrization, we perform a change of variables:

$$q(y | \theta) = \mathbb{E}[\mathbf{1}\{h_\theta(Z) = y\}]  = \int \mathbf{1} \{h_\theta(z) = y\} ~ r(z) ~ \mathrm{d}z$$

Assuming that $h_\theta$ is invertible, we take $z = h_\theta^{-1}(u)$ and

$$\mathrm{d}z = |\partial h_{\theta}^{-1}(u)| ~ \mathrm{d}u$$

so that

$$q(y | \theta) = \int \mathbf{1} \{u = y\} ~ r(h_\theta^{-1}(u)) ~ |\partial h_{\theta}^{-1}(u)| ~ \mathrm{d}u$$

We can now differentiate, but it gets tedious.

## Bibliography

```@bibliography
```