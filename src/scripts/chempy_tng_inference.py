import paths
import numpy as np

from plot_functions import *

# ----- Config -------------------------------------------------------------------------------------------------------------------------------------------
name = "NPE_C"

# --- Load simulated data ---
alpha_IMF_obs = np.load(paths.data / f"alpha_IMF_obs_{name}.npy")
log10_N_Ia_obs = np.load(paths.data / f"log10_N_Ia_obs_{name}.npy")

# --- Define the simulator params used for the data ---
# see sample_posterior.py for the definition of these parameters

simulations = 1000
N_stars = 1000
global_params = np.array([[-2.3, -2.89], [0.3, 0.3]])

# --- Multi-star inference ---
# Calculate mean and std for each observation
mu_alpha, sigma_alpha = alpha_IMF_obs.mean(axis=1), alpha_IMF_obs.std(axis=1)
mu_N_Ia, sigma_N_Ia = log10_N_Ia_obs.mean(axis=1), log10_N_Ia_obs.std(axis=1)

# Calculate mean and std for joint distribution
mu_alpha_combined = np.sum(mu_alpha/sigma_alpha**2)/np.sum(1/sigma_alpha**2)
sigma_alpha_combined = 1/np.sqrt(np.sum(1/sigma_alpha**2))

mu_N_Ia_combined = np.sum(mu_N_Ia/sigma_N_Ia**2)/np.sum(1/sigma_N_Ia**2)
sigma_N_Ia_combined = 1/np.sqrt(np.sum(1/sigma_N_Ia**2))

print(f"alpha_IMF = {mu_alpha_combined:.3f} +/- {sigma_alpha_combined:.3f}")
print(f"log10_N_Ia = {mu_N_Ia_combined:.3f} +/- {sigma_N_Ia_combined:.3f}")


# --- Plot the data ---
gaussian_posterior_plot(alpha_IMF_obs, log10_N_Ia_obs, global_params, title="CHEMPY_TNG_yields")


#stars = np.arange(1,1000)
#n_stars_plot(alpha_IMF_obs, log10_N_Ia_obs, global_params, "CHEMPY_TNG_yields_N_star", stars)


# --- Compare to HMC ---
philcox = {}
philcox["n_stars"] = [1,10,100]
philcox["med"] = [[-2.29,-2.87],[-2.31,-2.90],[-2.31,-2.90]]
philcox["up"] = [[-2.21,-2.76],[-2.29,-2.87],[-2.30,-2.89]]
philcox["lo"] = [[-2.37,-2.98],[-2.33,-2.93],[-2.32,-2.91]]

stars = np.arange(1,1000)
n_stars_plot_comp2(alpha_IMF_obs, log10_N_Ia_obs, global_params, philcox, "CHEMPY_TNG_yields_N_star_comp", stars)
