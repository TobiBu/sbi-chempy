import corner
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
import torch
from Chempy.parameter import ModelParameters
from numpyro.infer import MCMC, NUTS

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


# ---- NumPyro model using PyTorch emulator ----
def numpyro_model(obs_abundances, obs_errors):
    # Sample all parameters
    alpha_imf = numpyro.sample("alpha_imf", dist.Normal(-2.3, 0.3))
    log10_n_ia = numpyro.sample("log10_n_ia", dist.Normal(-2.89, 0.3))
    log10_sfe = numpyro.sample("log10_sfe", dist.Normal(-0.3, 0.3))
    log10_sfr_peak = numpyro.sample("log10_sfr_peak", dist.Normal(0.55, 0.1))
    xout = numpyro.sample("xout", dist.Normal(0.5, 0.1))
    birth_time = numpyro.sample("birth_time", dist.Uniform(1.0, 13.8))

    # Pack into a JAX DeviceArray and convert to a numpy float array explicitly using numpyro.deterministic
    # Note: The model is not traced through the emulator, so this is fine
    param_array = numpyro.deterministic(
        "input_array",
        jnp.array([alpha_imf, log10_n_ia, log10_sfe, log10_sfr_peak, xout, birth_time]),
    )

    # Convert to numpy explicitly here (OUTSIDE any JAX tracing context)
    param_numpy = np.asarray(param_array, dtype=np.float32)
    input_tensor = torch.from_numpy(param_numpy)

    # Run emulator and convert output
    predicted_abundances = model(input_tensor).detach().numpy()

    # Likelihood
    numpyro.sample(
        "obs", dist.Normal(predicted_abundances, obs_errors), obs=obs_abundances
    )


# ---- Run MCMC ----
nuts_kernel = NUTS(numpyro_model)
mcmc = MCMC(nuts_kernel, num_warmup=1000, num_samples=2000, num_chains=1)
rng_key = jax.random.PRNGKey(0)
mcmc.run(rng_key, obs_abundances=obs_abundances, obs_errors=obs_errors)
mcmc.print_summary()

samples = mcmc.get_samples()

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
