# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
# ---

# %% [markdown]
r"""
# State Space Model with Dynamic Coefficient $\alpha_t$ for Exponential Smoothing
**Coding Example**
"""
# %%
import pymc as pm
import numpy as np
import matplotlib.pyplot as plt

# %%
# Simulated data (replace with your own data)
n = 100  # Number of observations
true_alpha = 0.2  # True alpha value to generate data
x = np.zeros(n)
y = np.zeros(n)
e = np.random.normal(0, 2, n)  # Observation error
x[0] = np.random.normal(0, 1)  # Initial state

# Generate simulated data
for t in range(1, n):
    x[t] = x[t-1] + true_alpha * e[t]
    y[t] = x[t-1] + e[t]
# %%
with pm.Model() as model:
    # Define priors for unknown parameters.
    # Use a Beta distribution for alpha, with hyperparameters you 
    # can adjust as needed.
    
    # Dynamic coefficient alpha between 0 and 1
    alpha = pm.Beta('alpha', alpha=1, beta=1, shape=n)
    
    # Standard deviation of observation error
    sigma_e = pm.HalfNormal('sigma_e', sigma=1)  

    # Define the state space model
    
    # Initial state
    x_prior = pm.Normal('x_0', mu=0, sigma=2)  
    x_obs = pm.Normal('x_obs', mu=x_prior + alpha[:-1] * e[1:], sigma=sigma_e, observed=y[1:] - e[1:])

    # Sampling
    print("Starting sampling...")
    trace = pm.sample(
        1_000, 
        return_inferencedata=True, 
        progressbar=True
    )
    print("Sampling completed.")

# Results analysis
print(pm.summary(trace))

# %%
# Plot observations vs state
plt.plot(y, label='Observations $y_t$')
plt.plot(x, label='State $x_t$')
tmp = f"""
Observation vs. State

$y(t) = x(t-1) + e(t), e(t) \sim N(0, \sigma_e)$
$x(t) = x(t-1) + \\alpha(t) e(t)$
"""
plt.title(tmp)
plt.legend()

# %%
# Calculate mean alpha, observation error, and mean sigma_e
alpha_mean = trace.posterior['alpha'].values.mean(axis=(0,1))

error = y - x

sigma_e_samples = trace.posterior["sigma_e"].values

sigma_e_mean = sigma_e_samples.mean(axis=(0, 1))

sigma_error = (1 - alpha_mean) * sigma_e_mean

# %%
# Plot observation error with 1-sigma confidence bands
plt.plot(error, label='Observation Error $y_t - x_t$')
s_factor = 1
plt.fill_between(
    np.arange(n), 
    error - s_factor * sigma_error, error + s_factor * sigma_error,
    color='lightgray',
    label='$\\pm (1 - \\alpha_t)*\sigma_e$'
)
plt.legend()
plt.title('Observation Error vs. 1-sigma CI')
plt.axhline(0, color='black', lw=1, linestyle='--')

# %%
# Plot estimated alpha over time
plt.plot(alpha_mean, label='Estimated $\\alpha_t$')
plt.axhline(
    true_alpha,
    color='black',
    lw=1,
    linestyle='--',
    label='True $\\alpha_t$'
)
plt.legend()
plt.title('Estimated $\\alpha_t$ over time')
# %%
