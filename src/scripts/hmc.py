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


def log_prob_fn(params):
    # Convert to NumPy array if passed in as JAX array
    params = np.asarray(params)

    # Unpack parameters
    alpha_imf, log10_n_ia, log10_sfe, log10_sfr_peak, xout, birth_time = params

    # Reject unphysical samples
    if not (1.0 <= birth_time <= 13.8):
        return -np.inf

    # Gaussian priors
    logp = 0.0
    logp += -0.5 * ((alpha_imf + 2.3) / 0.3) ** 2
    logp += -0.5 * ((log10_n_ia + 2.89) / 0.3) ** 2
    logp += -0.5 * ((log10_sfe + 0.3) / 0.3) ** 2
    logp += -0.5 * ((log10_sfr_peak - 0.55) / 0.1) ** 2
    logp += -0.5 * ((xout - 0.5) / 0.1) ** 2

    # Run emulator (PyTorch)
    input_tensor = torch.tensor(params, dtype=torch.float32)
    with torch.no_grad():
        prediction = model(input_tensor).numpy()

    # Gaussian likelihood
    residual = (prediction - obs_abundances) / obs_errors
    log_likelihood = -0.5 * np.sum(residual**2)

    return logp + log_likelihood


# DO NOT JIT THIS — keep PyTorch isolated from JAX tracing
kernel = HMC(log_prob_fn)  # , step_size=0.01, num_steps=10)
mcmc = MCMC(kernel, num_warmup=500, num_samples=1000, num_chains=1)

# Initial parameter guess (safe values)
initial_params = np.array([-2.3, -2.89, -0.3, 0.55, 0.5, 6.0], dtype=np.float32)

# Run MCMC
mcmc.run(jax.random.PRNGKey(0), initial_params)

# Retrieve samples
samples = mcmc.get_samples()

samples_np = np.asarray(samples)
param_names = [
    "alpha_imf",
    "log10_n_ia",
    "log10_sfe",
    "log10_sfr_peak",
    "xout",
    "birth_time",
]

# figure = corner.corner(samples_np, labels=param_names, show_titles=True)
figure = corner.corner(
    samples_np,
    labels=param_names,
    show_titles=True,
    range=[(col.min() - 1e-2, col.max() + 1e-2) for col in samples_np.T],
)
plt.savefig(paths.figures / "hmc_corner_plot.pdf", dpi=300, bbox_inches="tight")
