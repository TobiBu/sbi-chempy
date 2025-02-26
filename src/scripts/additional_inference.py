import paths
import numpy as np
from chempy_torch_model import Model_Torch

import torch
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform
from scipy.stats import norm
from tqdm import tqdm
from Chempy.parameter import ModelParameters
import sbi.utils as utils
import time as t
import pickle
from plot_functions import *

# ----- Config -------------------------------------------------------------------------------------------------------------------------------------------
name = "NPE_C"

# --- Load simulated data ---
# ----- Load the data -----
a = ModelParameters()
labels_out = a.elements_to_trace
labels_in = [a.to_optimize[i] for i in range(len(a.to_optimize))] + ['time']
priors = torch.tensor([[a.priors[opt][0], a.priors[opt][1]] for opt in a.to_optimize])

elements = a.elements_to_trace

# ----- Load posterior -----
with open(paths.data / f'posterior_{name}.pickle', 'rb') as f:
    posterior = pickle.load(f)


# --- Set up the model ---
model = Model_Torch(len(labels_in), len(labels_out))

# --- Load the weights ---
model.load_state_dict(torch.load(paths.data / 'pytorch_state_dict.pt'))
model.eval()

# --- Set up the priors ---
local_GP = utils.MultipleIndependent(
    [Normal(p[0]*torch.ones(1), p[1]*torch.ones(1)) for p in priors[2:]] +
    [Uniform(torch.tensor([2.0]), torch.tensor([12.8]))],
    validate_args=False)

global_GP = utils.MultipleIndependent(
    [Normal(p[0]*torch.ones(1), p[1]*torch.ones(1)) for p in priors[:2]],
    validate_args=False)

# --- Simulate the data ---
N_stars = 1000
simulations = 1000

stars = local_GP.sample((N_stars,))

global_params_plot = np.array([[-2.1, -3.0], [0.3, 0.3]])
global_params = torch.tensor([[-2.1, -3.0]])


stars = torch.cat((global_params.repeat(N_stars, 1), stars), dim=1)

start = t.time()
abundances = model(stars)  
# Remove H from data, because it is just used for normalization (output with index 2)
abundances = torch.cat([abundances[:,0:2], abundances[:,3:]], axis=1)
end = t.time()
print(f'Time to create data for {N_stars} stars: {end-start:.3f} s')


def add_noise(true_abundances):
    # Define observational erorrs
    pc_ab = 5 # percentage error in abundance

    # Jitter true abundances and birth-times by these errors to create mock observational values.
    obs_ab_errors = np.ones_like(true_abundances)*float(pc_ab)/100.
    obs_abundances = norm.rvs(loc=true_abundances,scale=obs_ab_errors)

    return obs_abundances

alpha_IMF_obs = []
log10_N_Ia_obs = []
simulations = 1000

start = t.time()
for i in tqdm(range(len(abundances))):
    x = add_noise(abundances[i].detach().numpy())
    alpha, N_Ia = posterior.sample((simulations,), x=x, show_progress_bars=False)[:,0:2].T
    alpha_IMF_obs.append(alpha)
    log10_N_Ia_obs.append(N_Ia)
end = t.time()
print(f'Time to run {simulations} simulations for {N_stars} stars: {end-start:.3f} s')

alpha_IMF_obs = np.array(alpha_IMF_obs)
log10_N_Ia_obs = np.array(log10_N_Ia_obs)

gaussian_posterior_plot_n_stars(alpha_IMF_obs, log10_N_Ia_obs, global_params_plot, title="different_prior", philcox=None, no_stars=100)

# --- plot the data ---
#gaussian_posterior_plot(alpha_IMF_obs, log10_N_Ia_obs, global_params_plot, title="different_prior")

stars = np.arange(1,1000)
n_stars_plot(alpha_IMF_obs, log10_N_Ia_obs, global_params_plot, "Nstar_comp_different_prior", stars)