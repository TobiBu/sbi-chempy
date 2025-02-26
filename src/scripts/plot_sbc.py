import paths
import pickle
import torch
import numpy as np
from scipy.stats import norm
from Chempy.parameter import ModelParameters
from tqdm import tqdm
import matplotlib.pyplot as plt


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

theta_hat = torch.zeros_like(val_theta)
theta_predicted_stored = torch.zeros((len(abundances), 1000, len(labels_in)))
for index in tqdm(range(len(abundances))):
    thetas_predicted = posterior.sample((1000,), x=abundances[index], show_progress_bars=False)
    theta_predicted_stored[index] = thetas_predicted
    thetas_predicted_mean = thetas_predicted.mean(dim=0)
    theta_hat[index] = thetas_predicted_mean

# --- Plot calbration using ltu-ili ---
from metrics import PosteriorCoverage, PlotSinglePosterior

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

metric = PlotSinglePosterior(
    num_samples=1000, sample_method='direct', 
    labels=labels_in
)

# --- Plot corner plot posterior for single star using ltu-ili ---
fig = metric(
    posterior=posterior,
    x=abundances[0], theta=val_theta[0],
    plot_kws=dict(fill=True))

fig.savefig(paths.figures / 'corner_plot_singlestar.pdf')

# --- Plot the correlation between the distance and the correlation ---
correlation = np.zeros(len(abundances))
distance = np.zeros(len(abundances))
mahalanobis_distance = np.zeros(len(abundances))
for i in tqdm(range(len(abundances))):
    samples = theta_predicted_stored[i]
    correlation[i] = np.cov(samples[:, :2].T)[0, 1]
    distance_vec = val_theta[i][:2] - samples.mean(dim=0)[:2]
    distance[i] = np.linalg.norm(distance_vec)
    mahalanobis_distance[i] = np.sqrt( (np.array(distance_vec)).T @ np.linalg.inv(np.cov(np.array(samples[:, :2].T) )) @ (np.array(distance_vec)) )

percentile = np.percentile(correlation, [0, 99])
percentile_mask = (correlation>percentile[0])&(correlation<percentile[1])