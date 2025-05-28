import corner
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
import torch
from Chempy.parameter import ModelParameters
from numpyro.infer import HMC, MCMC, NUTS

import paths
from chempy_torch_model import Model_Torch

# ----- Load the data -----
a = ModelParameters()
labels_out = a.elements_to_trace
labels = [a.to_optimize[i] for i in range(len(a.to_optimize))] + ["time"]
priors = torch.tensor([[a.priors[opt][0], a.priors[opt][1]] for opt in a.to_optimize])


# ---- Load your emulator weights ----
model = Model_Torch(len(labels), len(labels_out))
model.load_state_dict(torch.load(paths.data / "pytorch_state_dict.pt"))
model.eval()

# ---- Load observational data from validation file ----
val_data = np.load(paths.data / "chempy_data/chempy_TNG_val_data.npz", mmap_mode="r")
val_theta = val_data["params"]
val_x = val_data["abundances"]


def clean_data(x, y):
    index = np.where((y == 0).all(axis=1))[0]
    x = np.delete(x, index, axis=0)
    y = np.delete(y, index, axis=0)
    index = np.where(np.isfinite(y).all(axis=1))[0]
    return x[index], y[index]


val_theta, val_x = clean_data(val_theta, val_x)
obs_abundances = val_x[0]
obs_errors = np.ones_like(obs_abundances) * 0.05  # fixed Gaussian noise


# Log-posterior
def log_prob_fn(params):
    # Extract parameters
    alpha_imf, log10_n_ia, log10_sfe, log10_sfr_peak, xout, birth_time = params

    # Priors
    logp = 0.0
    logp += -0.5 * ((alpha_imf + 2.3) / 0.3) ** 2
    logp += -0.5 * ((log10_n_ia + 2.89) / 0.3) ** 2
    logp += -0.5 * ((log10_sfe + 0.3) / 0.3) ** 2
    logp += -0.5 * ((log10_sfr_peak - 0.55) / 0.1) ** 2
    logp += -0.5 * ((xout - 0.5) / 0.1) ** 2
    if birth_time < 1.0 or birth_time > 13.8:
        return -jnp.inf

    # Emulator
    input_tensor = torch.tensor(
        [alpha_imf, log10_n_ia, log10_sfe, log10_sfr_peak, xout, birth_time],
        dtype=torch.float32,
    )
    with torch.no_grad():
        prediction = model(input_tensor).numpy()

    # Likelihood
    residual = (prediction - obs_abundances) / obs_errors
    log_likelihood = -0.5 * np.sum(residual**2)
    return logp + log_likelihood


# Wrap for JAX
def wrapped_logprob(q):
    return jax.lax.cond(
        jnp.logical_and(q[-1] >= 1.0, q[-1] <= 13.8),
        lambda _: log_prob_fn(q),
        lambda _: -jnp.inf,
        operand=None,
    )


# Initialize
initial_params = np.array([-2.3, -2.89, -0.3, 0.55, 0.5, 5.0], dtype=np.float32)

# Run HMC
kernel = HMC(wrapped_logprob, step_size=0.01, num_steps=10)
mcmc = MCMC(kernel, num_warmup=500, num_samples=1000)
mcmc.run(jax.random.PRNGKey(0), init_params=initial_params)
samples = mcmc.get_samples()

# Convert to NumPy
samples_np = np.array(samples)

# Convert the samples dict to a NumPy array for plotting
param_names = [
    "alpha_imf",
    "log10_n_ia",
    "log10_sfe",
    "log10_sfr_peak",
    "xout",
    "birth_time",
]
sample_array = np.vstack([samples[p].numpy() for p in param_names]).T

# Create corner plot
figure = corner.corner(
    sample_array,
    labels=param_names,
    truths=None,  # or provide [true_alpha_imf, true_log10_n_ia, ...] if known
    show_titles=True,
    title_fmt=".3f",
    title_kwargs={"fontsize": 12},
)

plt.savefig(paths.figures / "hmc_corner_plot.pdf", dpi=300, bbox_inches="tight")
