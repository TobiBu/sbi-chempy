import pickle
import time as t

import numpy as np
from scipy.stats import norm
from tqdm import tqdm

import paths
from plot_functions import *

# ----- Config -------------------------------------------------------------------------------------------------------------------------------------------
name = "NPE_C"

global_params = np.array([[-2.3, -2.89], [0.3, 0.3]])


# --- Clean the data ---
# Chempy sometimes returns zeros or infinite values, which need to removed
def clean_data(x, y):
    # Remove all zeros from the training data
    index = np.where((y == 0).all(axis=1))[0]
    x = np.delete(x, index, axis=0)
    y = np.delete(y, index, axis=0)

    # Remove all infinite values from the training data
    index = np.where(np.isfinite(y).all(axis=1))[0]
    x = x[index]
    y = y[index]

    return x, y


# ----- Add noise to data -------------------------------------------------------------------------------------------------------------------------------------------
def add_noise(true_abundances):
    # Define observational erorrs
    pc_ab = 5  # percentage error in abundance

    # Jitter true abundances and birth-times by these errors to create mock observational values.
    obs_ab_errors = np.ones_like(true_abundances) * float(pc_ab) / 100.0
    obs_abundances = norm.rvs(loc=true_abundances, scale=obs_ab_errors)

    return obs_abundances


# ----- Load the data -------------------------------------------------------------------------------------------------------------------------------------------
# CHEMPY data with alternative yields
data_alternative_yields = np.load(
    paths.data / "chempy_data/chempy_alternative_yields.npz", mmap_mode="r"
)

elements = data_alternative_yields["elements"]
alt_x = data_alternative_yields["params"]
alt_y = data_alternative_yields["abundances"]

alt_x, alt_y = clean_data(alt_x, alt_y)

# Remove H from data, because it is just used for normalization (output with index 2)
elements = np.delete(elements, 2)
alt_y = np.delete(alt_y, 2, 1)

alt_y_obs = add_noise(alt_y)

# ----- Load posterior -----
with open(paths.data / f"posterior_{name}.pickle", "rb") as f:
    posterior = pickle.load(f)

# ----- Evaluate the posterior -------------------------------------------------------------------------------------------------------------------------------------------
alpha_IMF_alt_obs = []
log10_N_Ia_alt_obs = []
simulations = 1000
N_stars = len(alt_y_obs)

start = t.time()
for i in tqdm(range(len(alt_y_obs))):
    x = add_noise(alt_y_obs[i])
    alpha, N_Ia = posterior.sample((simulations,), x=x, show_progress_bars=False)[
        :, 0:2
    ].T
    alpha_IMF_alt_obs.append(alpha)
    log10_N_Ia_alt_obs.append(N_Ia)
end = t.time()
print(f"Time to run {simulations} simulations for {N_stars} stars: {end-start:.3f} s")

alpha_IMF_alt_obs = np.array(alpha_IMF_alt_obs)
log10_N_Ia_alt_obs = np.array(log10_N_Ia_alt_obs)


# ------ plot the data -------------------------------------------------------------------------------------------------------------------------------------------
# --- Compare to HMC ---
# Philcox&Rybizki 2019 Table 3
philcox = {}
philcox["n_stars"] = np.array([1, 10, 100])
philcox["med"] = np.array([[-2.25, -3.01], [-2.21, -2.96], [-2.22, -2.96]])
philcox["up"] = np.array([[-2.14, -2.86], [-2.17, -2.88], [-2.20, -2.93]])
philcox["lo"] = np.array([[-2.34, -3.16], [-2.25, -3.04], [-2.24, -2.98]])

gaussian_posterior_plot_n_stars(
    alpha_IMF_alt_obs,
    log10_N_Ia_alt_obs,
    global_params,
    title="CHEMPY_alternative_yields",
    philcox=None,
    no_stars=100,
)

stars = np.arange(1, len(alt_y_obs))
n_stars_plot_comp2(
    alpha_IMF_alt_obs,
    log10_N_Ia_alt_obs,
    global_params,
    philcox,
    "CHEMPY_alternative_yields_Nstar_comp",
    stars,
)

# gaussian_posterior_plot(alpha_IMF_alt_obs, log10_N_Ia_alt_obs, global_params, title="CHEMPY_alternative_yields")
