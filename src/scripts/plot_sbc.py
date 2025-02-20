import paths
import pickle
import torch
import numpy as np
from scipy.stats import norm
from Chempy.parameter import ModelParameters


# ----- Config -------------------------------------------------------------------------------------------------------------------------------------------
name = "NPE_C"

# --- Define the prior ---
a = ModelParameters()
labels_out = a.elements_to_trace
labels_in = [a.to_optimize[i] for i in range(len(a.to_optimize))] + ['time']

# ----- load the posterior -------------------------------------------------------------------------------------------------------------------------------------------
with open(paths.data / f'posterior_{name}.pickle', 'rb') as f:
    posterior = pickle.load(f)

# --- Load the validation data ---
# Validation data created with CHEMPY, not with the NN simulator
print("Evaluating the posterior...")
path_test = paths.data / 'chempy_data/chempy_TNG_val_data.npz'
val_data = np.load(path_test, mmap_mode='r')

val_theta = val_data['params']
val_x = val_data['abundances']

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

val_theta, val_x = clean_data(val_theta, val_x)

val_theta = val_theta[:5000]
val_x = val_x[:5000]

# convert to torch tensors
val_theta = torch.tensor(val_theta, dtype=torch.float32)
val_x = torch.tensor(val_x, dtype=torch.float32)
abundances =  torch.cat([val_x[:,:2], val_x[:,3:]], dim=1)

# --- add noise ---
pc_ab = 5 # percentage error in abundance

# add noise to data to simulate observational errors
x_err = np.ones_like(abundances)*float(pc_ab)/100.
abundances = norm.rvs(loc=abundances,scale=x_err)
abundances = torch.tensor(abundances).float()

# --- Plot calbration using ltu-ili ---
from metrics import PosteriorCoverage

plot_hist = ["coverage", "histogram", "predictions", "tarp"]
metric = PosteriorCoverage(
    num_samples=1000, sample_method='direct',
    labels=labels_in,
    plot_list = plot_hist
)

fig = metric(
    posterior=posterior,
    x=abundances, theta=val_theta)

for i, plot in enumerate(fig):
    fig[i].savefig(paths.figures / f'ili_{plot_hist[i]}.pdf')