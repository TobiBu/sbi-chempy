import pickle

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
obs_abundances = val_x[:50]
val_x = val_x[:50]
val_theta = val_theta[:50]
obs_errors = np.ones_like(obs_abundances) * 0.05  # fixed Gaussian noise


def make_log_prob_fn(obs_abundances, obs_errors):
    def log_prob_fn(params):
        params = np.asarray(params)
        alpha_imf, log10_n_ia, log10_sfe, log10_sfr_peak, xout, birth_time = params

        if not (1.0 <= birth_time <= 13.8):
            return -np.inf

        logp = 0.0
        logp += -0.5 * ((alpha_imf + 2.3) / 0.3) ** 2
        logp += -0.5 * ((log10_n_ia + 2.89) / 0.3) ** 2
        logp += -0.5 * ((log10_sfe + 0.3) / 0.3) ** 2
        logp += -0.5 * ((log10_sfr_peak - 0.55) / 0.1) ** 2
        logp += -0.5 * ((xout - 0.5) / 0.1) ** 2

        input_tensor = torch.tensor(params, dtype=torch.float32)
        with torch.no_grad():
            prediction = model(input_tensor).numpy()

        if not np.all(np.isfinite(prediction)):
            return -np.inf

        residual = (prediction - obs_abundances) / obs_errors
        log_likelihood = -0.5 * np.sum(residual**2)

        return logp + log_likelihood

    return log_prob_fn


# === MH Sampler ===
def metropolis_hastings(log_prob_fn, initial, num_samples=1000, proposal_scale=0.05):
    current = np.array(initial)
    samples = []
    accepted = 0

    current_log_prob = log_prob_fn(current)

    for i in range(num_samples):
        proposal = current + np.random.normal(scale=proposal_scale, size=current.shape)
        proposal_log_prob = log_prob_fn(proposal)

        if np.log(np.random.rand()) < proposal_log_prob - current_log_prob:
            current = proposal
            current_log_prob = proposal_log_prob
            accepted += 1

        samples.append(current.copy())

        if (i + 1) % 100 == 0:
            print(f"Step {i+1}: acceptance rate = {accepted / (i+1):.2f}")

    return np.array(samples)


# Initial parameter guess (safe values)
initial_params = np.array([-2.3, -2.89, -0.3, 0.55, 0.5, 6.0], dtype=np.float32)

# print("logp(init):", log_prob_fn(initial_params))

mh_samples = []
obs_errors = np.ones(val_x.shape[1]) * 0.05

for i in range(len(val_x)):
    print(f"Sampling star {i+1}/{len(val_x)}")

    obs = val_x[i]
    truth = val_theta[i]

    log_prob = make_log_prob_fn(obs, obs_errors)
    initial = np.array(
        [-2.3, -2.89, -0.3, 0.55, 0.5, 6.0]
    )  # or truth as a starting point

    samples = metropolis_hastings(
        log_prob, initial, num_samples=2000, proposal_scale=0.02
    )
    mh_samples.append(
        {
            "samples": samples,
            "truth": truth,
        }
    )


samples_np = np.asarray(samples)
param_names = [
    "alpha_imf",
    "log10_n_ia",
    "log10_sfe",
    "log10_sfr_peak",
    "xout",
    "birth_time",
]

# for i, result in enumerate(mh_samples):
#    samples = result["samples"]
#    truth = result["truth"]

#    print(f"Plotting star {i}")
#    fig = corner.corner(
#        samples, labels=param_names, truths=truth, show_titles=True, title_fmt=".2f"
#    )
#    fig.suptitle(f"Star {i}")
#    plt.show()

with open("mh_results.pkl", "wb") as f:
    pickle.dump(mh_samples, f)


# Step 1: Stack samples from all stars
all_samples = np.vstack([d["samples"] for d in mh_samples])

# Optional: Thin if needed
# all_samples = all_samples[::10]

# Step 2: Plot the combined posterior
param_names = [
    "alpha_imf",
    "log10_n_ia",
    "log10_sfe",
    "log10_sfr_peak",
    "xout",
    "birth_time",
]

true_mean = np.mean(val_theta, axis=0)

fig = corner.corner(
    all_samples, labels=param_names, show_titles=True, truths=true_mean, title_fmt=".2f"
)
fig.suptitle("Global posterior (combined from all stars)")
plt.show()
