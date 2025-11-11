import pymc as pm
import numpy as np
import arviz as az
import matplotlib.pyplot as plt

print(f"Running on PyMC v{pm.__version__}")

# --- 1. Generate Synthetic Data ---
# This is the data we would have in a real-world problem.

N_points = 50
np.random.seed(42)

# Define true parameters
a_true = 2.5
b_true = 1.0

# Define the *true* unknown x values
x_true_data = np.linspace(0, 10, N_points)
y_true_data = a_true * x_true_data + b_true

# (IMPORTANT) Define HETEROSCEDASTIC errors (different for each point)
sigma_x_array = np.random.uniform(0.2, 0.8, size=N_points)
sigma_y_array = np.random.uniform(0.5, 1.5, size=N_points)

# Generate our *observed* data by adding noise
x_obs = np.random.normal(x_true_data, sigma_x_array)
y_obs = np.random.normal(y_true_data, sigma_y_array)

# --- 2. Define the Bayesian EIV Model ---

with pm.Model() as eiv_model:
    # --- Priors for Model Parameters (a, b) ---
    # Weakly informative priors
    a = pm.Normal('a', mu=0, sigma=10)
    b = pm.Normal('b', mu=0, sigma=10)
    
    # --- Prior for Latent Variable (x_true) ---
    # This is the core of the EIV model.
    # We define x_true as a latent variable. Its prior is centered
    # on our observed x_obs, with the known x_errors.
    #
    # We pass the *array* sigma_x_array to the sigma argument.
    # shape=N_points tells PyMC to create N_points independent x_true variables.
    x_true = pm.Normal('x_true', mu=x_obs, sigma=sigma_x_array, shape=N_points)
    
    # --- Model Definition ---
    # The deterministic model relates the *true* quantities
    mu_y = a * x_true + b
    
    # --- Likelihood (Y-Error Model) ---
    # This connects the model to our observed y_obs,
    # using our known y_errors (sigma_y_array).
    y_likelihood = pm.Normal('y_likelihood', 
                             mu=mu_y, 
                             sigma=sigma_y_array, 
                             observed=y_obs)

# --- 3. Run the MCMC Sampler ---
with eiv_model:
    trace = pm.sample(2000, tune=1000, cores=4)

# --- 4. Analyze Results ---
print("Model fitting complete. Displaying summary for 'a' and 'b':")

# Show summary statistics for 'a' and 'b'
summary = az.summary(trace, var_names=['a', 'b'])
print(summary)
# The 94% HDI (Highest Density Interval) for 'a' and 'b' should contain
# the true values (a_true=2.5, b_true=1.0)

# --- 5. Visualize Results (Optional) ---
fig, axes = az.plot_trace(trace, var_names=['a', 'b'],
                          figsize=(14, 7))
plt.tight_layout(pad=2.0)
plt.savefig('trace_plot.png', dpi=150, bbox_inches='tight')

# Plot the fit against the data
plt.figure(figsize=(10, 6))
# Plot observed data with error bars
plt.errorbar(x_obs, y_obs, xerr=sigma_x_array, yerr=sigma_y_array, 
             fmt='o', alpha=0.5, label='Observed Data (with errors)')

# Plot the true line
x_line = np.linspace(-1, 11, 100)
plt.plot(x_line, a_true * x_line + b_true, 'r--', label='True Relationship')

# Plot posterior predictive fits
# Extract posterior samples for a and b
posterior_a = trace.posterior['a'].values.flatten()
posterior_b = trace.posterior['b'].values.flatten()

# Randomly sample 100 posterior draws
n_samples = 100
indices = np.random.choice(len(posterior_a), size=n_samples, replace=False)

# Plot each posterior fit line
for idx in indices:
    y_line = posterior_a[idx] * x_line + posterior_b[idx]
    plt.plot(x_line, y_line, 'C1', alpha=0.05)

plt.title('Bayesian EIV Fit with PyMC', fontsize=14, fontweight='bold')
plt.xlabel('X', fontsize=12)
plt.ylabel('Y', fontsize=12)
plt.legend(loc='best', frameon=True, fancybox=True, shadow=True,
           framealpha=0.95, facecolor='white', edgecolor='gray', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig('eiv_fit.png', dpi=150, bbox_inches='tight')
