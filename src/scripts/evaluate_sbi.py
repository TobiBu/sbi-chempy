import paths
import pickle
import torch 
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform

import sbi.utils as utils
from sbi.analysis.plot import sbc_rank_plot, plot_tarp
from sbi.diagnostics import check_tarp, run_sbc, run_tarp

from Chempy.parameter import ModelParameters

from chempy_torch_model import Model_Torch
from plot_functions import *

# ----- Config -------------------------------------------------------------------------------------------------------------------------------------------

name = "NPE_C"

# --- Absolute percentage error plot ---

# --- Define the prior ---
a = ModelParameters()
labels_out = a.elements_to_trace
labels_in = [a.to_optimize[i] for i in range(len(a.to_optimize))] + ['time']
priors = torch.tensor([[a.priors[opt][0], a.priors[opt][1]] for opt in a.to_optimize])

combined_priors = utils.MultipleIndependent(
    [Normal(p[0]*torch.ones(1), p[1]*torch.ones(1)) for p in priors] +
    [Uniform(torch.tensor([2.0]), torch.tensor([12.8]))],
    validate_args=False)

# ----- load the posterior -------------------------------------------------------------------------------------------------------------------------------------------
with open(paths.data / f'posterior_{name}.pickle', 'rb') as f:
    posterior = pickle.load(f)

# --- Set up the model ---
model = Model_Torch(len(labels_in), len(labels_out))

# --- Load the weights ---
model.load_state_dict(torch.load(paths.data / 'pytorch_state_dict.pt'))
model.eval()

# --- Simulation based calibration plot ---
def simulator(params):
    y = model(params)
    y = y.detach().numpy()

    # Remove H from data, because it is just used for normalization (output with index 2)
    y = np.delete(y, 2,1)

    return y

num_sbc_samples = 200  # choose a number of sbc runs, should be ~100s
# generate ground truth parameters and corresponding simulated observations for SBC.
thetas = combined_priors.sample((num_sbc_samples,))
xs = simulator(thetas)

# run SBC: for each inference we draw 1000 posterior samples.
num_posterior_samples = 1_000
num_workers = 1
ranks, dap_samples = run_sbc(
    thetas, xs, posterior, num_posterior_samples=num_posterior_samples, num_workers=num_workers
)

# --- SBC rank plot ---
f, ax = sbc_rank_plot(
    ranks=ranks,
    num_posterior_samples=num_posterior_samples,
    parameter_labels=labels_in,
    plot_type="hist",
    num_cols=3,
    figsize=(15,10),
    num_bins=None,  # by passing None we use a heuristic for the number of bins.
)

f.suptitle("SBC rank plot", fontsize=36)
plt.tight_layout()
plt.savefig(paths.figures / f'sbc_rank_plot_{name}.pdf')
plt.clf()

# --- TARP plot ---
# the tarp method returns the ECP values for a given set of alpha coverage levels.
ecp, alpha = run_tarp(
    thetas,
    xs,
    posterior,
    references=None,  # will be calculated automatically.
    num_posterior_samples=1000,
)

# Similar to SBC, we can check then whether the distribution of ecp is close to
# that of alpha.
atc, ks_pval = check_tarp(ecp, alpha)
print(atc, "Should be close to 0")
print(ks_pval, "Should be larger than 0.05")

# Or, we can perform a visual check.
f, ax = plot_tarp(ecp, alpha)
plt.tight_layout()
plt.savefig(paths.figures / f'tarp_plot_{name}.pdf')
plt.clf()