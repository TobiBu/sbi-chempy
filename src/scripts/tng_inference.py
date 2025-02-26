import paths
import numpy as np

from scipy.stats import norm
from tqdm import tqdm

import time as t
import pickle

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
    pc_ab = 5 # percentage error in abundance

    # Jitter true abundances and birth-times by these errors to create mock observational values.
    obs_ab_errors = np.ones_like(true_abundances)*float(pc_ab)/100.
    obs_abundances = norm.rvs(loc=true_abundances,scale=obs_ab_errors)

    return obs_abundances

# ----- Load the data -------------------------------------------------------------------------------------------------------------------------------------------
# TNG simulation data
data_tng = np.load(paths.static / "Mock_Data_TNG.npz", mmap_mode='r')

tng_y = data_tng['true_abuns']

tng_y_obs = add_noise(tng_y)

# ----- Load posterior -----
with open(paths.data / f'posterior_{name}.pickle', 'rb') as f:
    posterior = pickle.load(f)

# ----- Evaluate the posterior -------------------------------------------------------------------------------------------------------------------------------------------
alpha_IMF_tng_obs = []
log10_N_Ia_tng_obs = []
simulations = 1000
N_stars = len(tng_y_obs)

start = t.time()
for i in tqdm(range(len(tng_y_obs))):
    x = add_noise(tng_y_obs[i])
    alpha, N_Ia = posterior.sample((simulations,), x=x, show_progress_bars=False)[:,0:2].T
    alpha_IMF_tng_obs.append(alpha)
    log10_N_Ia_tng_obs.append(N_Ia)
end = t.time()
print(f'Time to run {simulations} simulations for {N_stars} stars: {end-start:.3f} s')

alpha_IMF_tng_obs = np.array(alpha_IMF_tng_obs)
log10_N_Ia_tng_obs = np.array(log10_N_Ia_tng_obs)


# ------ plot the data -------------------------------------------------------------------------------------------------------------------------------------------
# --- Compare to HMC ---
# Philcox&Rybizki 2019 Table 3
philcox = {}
philcox["n_stars"] = np.array([1,10,100])
philcox["med"] = np.array([[-2.27,-2.86],[-2.27,-2.87],[-2.28,-2.89]])
philcox["up"] = np.array([[-2.19,-2.75],[-2.24,-2.84],[-2.27,-2.88]])
philcox["lo"] = np.array([[-2.35,-2.97],[-2.3,-2.91],[-2.29,-2.90]])

gaussian_posterior_plot_n_stars(alpha_IMF_tng_obs, log10_N_Ia_tng_obs, global_params, title="TNG_simulation_philcox", philcox=philcox, no_stars=100)

stars = np.arange(1,len(tng_y_obs))
n_stars_plot_comp2(alpha_IMF_tng_obs, log10_N_Ia_tng_obs, global_params, philcox, "TNG_sim_Nstar_comp", stars)

#gaussian_posterior_plot(alpha_IMF_tng_obs, log10_N_Ia_tng_obs, global_params, title="TNG_simulation")