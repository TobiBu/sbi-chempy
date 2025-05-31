import pickle
import time as t

import corner
import matplotlib.pyplot as plt
import numpy as np
import sbi.utils as utils
import torch
from Chempy.parameter import ModelParameters
from scipy.stats import norm
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform
from tqdm import tqdm

import paths
from chempy_torch_model import Model_Torch

# ----- Config -------------------------------------------------------------------------------------------------------------------------------------------
name = "NPE_C"

# ----- Load the data -----
a = ModelParameters()
labels_out = a.elements_to_trace
labels = [a.to_optimize[i] for i in range(len(a.to_optimize))] + ["time"]
priors = torch.tensor([[a.priors[opt][0], a.priors[opt][1]] for opt in a.to_optimize])

# --- Define the priors ---
local_GP = utils.MultipleIndependent(
    [Normal(p[0] * torch.ones(1), p[1] * torch.ones(1)) for p in priors[2:]]
    + [Uniform(torch.tensor([2.0]), torch.tensor([12.8]))],
    validate_args=False,
)

global_GP = utils.MultipleIndependent(
    [Normal(p[0] * torch.ones(1), p[1] * torch.ones(1)) for p in priors[:2]],
    validate_args=False,
)

# ----- load the posterior -------------------------------------------------------------------------------------------------------------------------------------------
with open(paths.data / f"posterior_{name}.pickle", "rb") as f:
    posterior = pickle.load(f)

# ---- Load your emulator weights ----
model = Model_Torch(len(labels), len(labels_out))
model.load_state_dict(torch.load(paths.data / "pytorch_state_dict.pt"))
model.eval()

# --- Define the simulator ---
N_stars = 200
N_samples = 1000

stars = local_GP.sample((N_stars,))
global_params = torch.tensor([[-2.3, -2.89]])

stars = torch.cat((global_params.repeat(N_stars, 1), stars), dim=1)

# ----- Simulate abundances -----
start = t.time()
abundances = model(stars)
# Remove H from data, because it is just used for normalization (output with index 2)
abundances = torch.cat([abundances[:, 0:2], abundances[:, 3:]], axis=1)
end = t.time()
print(f"Time to create data for {N_stars} stars: {end-start:.3f} s")


# ----- Add noise -----
def add_noise(true_abundances):
    # Define observational erorrs
    pc_ab = 5  # percentage error in abundance

    # Jitter true abundances and birth-times by these errors to create mock observational values.
    obs_ab_errors = np.ones_like(true_abundances) * float(pc_ab) / 100.0
    obs_abundances = norm.rvs(loc=true_abundances, scale=obs_ab_errors)

    return obs_abundances, obs_ab_errors


def clean_data(x, y):
    index = np.where((y == 0).all(axis=1))[0]
    x = np.delete(x, index, axis=0)
    y = np.delete(y, index, axis=0)
    index = np.where(np.isfinite(y).all(axis=1))[0]
    return x[index], y[index]


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
        logp += -np.log(12.8 - 2.0)  # log(1 / width)

        input_tensor = torch.tensor(params, dtype=torch.float32)
        with torch.no_grad():
            prediction = model(input_tensor).numpy()
            # Remove H from data, because it is just used for normalization (output with index 2)
            prediction = np.concatenate(([prediction[0:2], prediction[3:]]))

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

start = t.time()
for i in tqdm(range(len(abundances))):
    print(f"Sampling star {i+1}/{len(abundances)}")
    x, obs_errors = add_noise(abundances[i].detach().numpy())

    truth = stars[i]

    log_prob = make_log_prob_fn(x, obs_errors)
    initial = np.array(
        [-2.3, -2.89, -0.3, 0.55, 0.5, 6.0]
    )  # or truth as a starting point

    samples = metropolis_hastings(
        log_prob, initial, num_samples=N_samples, proposal_scale=0.02
    )
    mh_samples.append(
        {
            "samples": samples,
            "truth": truth,
        }
    )
end = t.time()
print(f"Time to run {samples} MH inferences for {N_stars} stars: {end-start:.3f} s")


samples_np = np.asarray(samples)
param_names = [
    "alpha_imf",
    "log10_n_ia",
    "log10_sfe",
    "log10_sfr_peak",
    "xout",
    "birth_time",
]

labels_in = [
    r"$\alpha_{\text{IMF}}$",
    r"$\log_{10}{\text{N}_{\text{Ia}}}$",
    r"$\log_{10}{\text{SFE}}$",
    r"$\log_{10}{\text{SFR}_{\text{peak}}}$",
    r"$x_{\text{out}}$",
    r"$\text{Time}$",
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

# --- Plot the posterior for MH vs. SBI ---

import seaborn as sns

# --- Plot calbration using ltu-ili ---
from metrics import PlotSinglePosterior

# Get two distinct colors
palette = sns.color_palette("colorblind", 2)
color_sbi = palette[0]
color_mh = palette[1]


plotter = PlotSinglePosterior(
    labels=labels_in,
    num_samples=1000,
    sample_method="direct",
)

sbi_samples = []

for k in range(len(abundances)):
    fig, data = plotter(
        posterior=posterior,
        x=abundances[k].detach().numpy(),
        theta=stars[k].detach().numpy(),
        plot_kws=dict(fill=True),
        mh_samples=mh_samples[k]["samples"],
        plot_kws_per_model={
            "SBI": dict(levels=[0.05, 0.32, 1], color=color_sbi, fill=True, alpha=0.6),
            "MH": dict(levels=[0.05, 0.32, 1], color=color_mh, fill=True, alpha=0.4),
        },
    )
    sbi_samples.append(data)
    fig.savefig(paths.figures / f"corner_plot_comparison_singlestar_{k}.pdf")

all_sbi_samples = np.vstack([d["samples"] for d in sbi_samples])

true_mean = np.mean(stars.numpy(), axis=0)
print("True mean parameters:", true_mean)
fig = corner.corner(
    all_sbi_samples,,
    labels=param_names,
    color="C0",
    bins=30,
    smooth=1.0,
    fill_contours=True,
    plot_density=True,
    plot_contours=True,
    show_titles=True,
    truths=true_mean,
    title_fmt=".2f",
)
# fig.suptitle("Global posterior (combined from all stars)")
# plt.show()

corner.corner(
    all_samples,  # shape: [N_samples, D]
    labels=param_names,
    fig=fig,  # reuse the same figure
    color="C1",
    bins=30,
    smooth=1.0,
    fill_contours=False,  # outlines only
    plot_density=True,
    plot_contours=True,
    hist_kwargs={"linestyle": "--", "linewidth": 1.5},
    contour_kwargs={"linestyle": "--", "linewidth": 1.5},
    no_fill_contours=True,  # keep SBI shaded, MH outlined
)

plt.savefig(paths.figure / "mh_results.pdf", dpi=300, bbox_inches="tight")
