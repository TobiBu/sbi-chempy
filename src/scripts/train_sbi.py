import pickle
import time as t

import numpy as np
import sbi.utils as utils
import torch
from Chempy.parameter import ModelParameters
from sbi.inference import NPE_C, simulate_for_sbi
from sbi.neural_nets import posterior_nn
from sbi.utils.user_input_checks import (
    check_sbi_inputs,
    process_prior,
    process_simulator,
)
from scipy.stats import norm
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform

import paths
from chempy_torch_model import Model_Torch

# ----- Config -------------------------------------------------------------------------------------------------------------------------------------------

name = "NPE_C"

# ----- Load the model -------------------------------------------------------------------------------------------------------------------------------------------
# --- Define the prior ---
a = ModelParameters()
labels_out = a.elements_to_trace
labels_in = [a.to_optimize[i] for i in range(len(a.to_optimize))] + ["time"]
priors = torch.tensor([[a.priors[opt][0], a.priors[opt][1]] for opt in a.to_optimize])


combined_priors = utils.MultipleIndependent(
    [Normal(p[0] * torch.ones(1), p[1] * torch.ones(1)) for p in priors]
    + [Uniform(torch.tensor([2.0]), torch.tensor([12.8]))],
    validate_args=False,
)

"""
combined_priors = utils.MultipleIndependent(
    [Uniform(p[0]*torch.ones(1)-5*p[1], p[0]*torch.ones(1)+5*p[1]) for p in priors] +
    [Uniform(torch.tensor([2.0]), torch.tensor([12.8]))],
    validate_args=False)
"""


# --- Set up the model ---
model = Model_Torch(len(labels_in), len(labels_out))

# --- Load the weights ---
model.load_state_dict(torch.load(paths.data / "pytorch_state_dict.pt"))
model.eval()


# ----- Set up the simulator -------------------------------------------------------------------------------------------------------------------------------------------
def simulator(params):
    y = model(params)
    y = y.detach().numpy()

    # Remove H from data, because it is just used for normalization (output with index 2)
    y = np.delete(y, 2)

    return y


prior, num_parameters, prior_returns_numpy = process_prior(combined_priors)
simulator = process_simulator(simulator, prior, prior_returns_numpy)
check_sbi_inputs(simulator, prior)


# ----- Train the SBI -------------------------------------------------------------------------------------------------------------------------------------------
density_estimator_build_fun = posterior_nn(
    model="maf", hidden_features=8, num_transforms=4
)  # , blocks=1)
inference = NPE_C(
    prior=prior, density_estimator=density_estimator_build_fun, show_progress_bars=True
)

start = t.time()

# --- simulate the data ---
print()
print("Simulating data...")
theta, x = simulate_for_sbi(simulator, proposal=prior, num_simulations=100_000)
print(f"Genereted {len(theta)} samples")

# --- add noise ---
pc_ab = 5  # percentage error in abundance

x_err = np.ones_like(x) * float(pc_ab) / 100.0
x = norm.rvs(loc=x, scale=x_err)
x = torch.tensor(x).float()

# --- train ---
print()
print("Training the posterior...")
density_estimator = inference.append_simulations(theta, x).train(
    show_train_summary=True
)

# --- build the posterior ---
posterior = inference.build_posterior(density_estimator)

end = t.time()
comp_time = end - start

print()
print(
    f"Time taken to train the posterior with {len(theta)} samples: "
    f'{np.floor(comp_time/60).astype("int")}min {np.floor(comp_time%60).astype("int")}s'
)


# ----- Save the posterior -------------------------------------------------------------------------------------------------------------------------------------------
with open(paths.data / f"posterior_{name}.pickle", "wb") as f:
    pickle.dump(posterior, f)

print()
print("Posterior trained and saved!")
print()
