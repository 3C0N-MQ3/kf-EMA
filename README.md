# State Space Model with Dynamic Coefficient $\alpha_t$ for Exponential Smoothing

This document outlines a state space model designed for analyzing time series data, incorporating a dynamic coefficient $\alpha_t$ to offer a flexible approach to exponential smoothing. This innovative model adapts to the evolving relationships between consecutive observations by allowing the smoothing coefficient to change over time.

## Model Description

The state space model consists of two primary components:

1. **Observation Equation**: It illustrates the relationship between the current observation $y_t$ and the previous state $x_{t-1}$, alongside an error term $e_t$:

$$
y_t = x_{t-1} + e_t, \quad e_t \sim N(0, \sigma^2_e)
$$

2. **State Transition Equation**: This equation defines the evolution of the state $x_t$ from $x_{t-1}$, incorporating the error term modified by the dynamically varying coefficient $\alpha_t$:

$$
x_t = x_{t-1} + \alpha_t e_t
$$

The dynamic coefficient $\alpha_t$ is modeled at each time point using a Beta distribution, ensuring its values remain within the [0, 1] interval. This setup captures the essence of exponential smoothing, with the added benefit of dynamic adaptability.

### Standard Deviation of Observation Error

The standard deviation of the difference between observation and state, $y_t - x_t$, is expressed as $(1 - \alpha_t)\sigma_e$. This formulation indicates how the uncertainty in the observation error scales with the value of $\alpha_t$, directly linking the model's adaptability to its predictive confidence.

## Assumptions

- **Observation Error**: The error $e_t$ is assumed to be a Gaussian process with zero mean and known variance $\sigma^2_e$, reflecting the model's uncertainty about observations.
- **Dynamic Coefficient $\alpha_t$**: Varies over time within the [0, 1] interval, serving as an adaptive smoothing factor that modulates the influence of past observations on current state predictions.
- **Independence**: Error terms at different points in time are assumed to be independent, simplifying the model's structure.

## Bayesian Estimation Process

Employing PyMC for Bayesian estimation, the model leverages prior distributions for unknown parameters, refining these priors into posterior distributions based on observed data:

1. **Prior Definitions**:
   - A Beta distribution is chosen for $\alpha_t$, suitable for coefficients that are naturally bounded between 0 and 1.
   - The standard deviation of the observation error, $\sigma_e$, is modeled using a HalfNormal distribution, emphasizing its positive value constraint.

2. **Sampling**:
   - Monte Carlo Markov Chain (MCMC) sampling is used to generate posterior distributions for the model parameters, facilitated by PyMC's efficient algorithms and diagnostic tools.

## Usage of PyMC

PyMC, a powerful tool for Bayesian statistical modeling and probabilistic inference, provides:

- Intuitive model definition syntax.
- Efficient MCMC sampling.
- Diagnostic tools for assessing sampling convergence.
- Statistical summaries of sampling results.

This approach allows for a nuanced and adaptable estimation of complex time series models, highlighting the integration of uncertainty and dynamic adaptability in the context of exponential smoothing.
