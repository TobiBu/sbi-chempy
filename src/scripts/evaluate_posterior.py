import pickle

import numpy as np
import torch
from Chempy.parameter import ModelParameters
from scipy.stats import norm
from tqdm import tqdm

import paths

# ----- Evaluate the posterior -------------------------------------------------------------------------------------------------------------------------------------------

# ----- Config -------------------------------------------------------------------------------------------------------------------------------------------
name = "NPE_C"

# --- Define the prior ---
a = ModelParameters()
labels_out = a.elements_to_trace
labels_in = [a.to_optimize[i] for i in range(len(a.to_optimize))] + ["time"]

# ----- load the posterior -------------------------------------------------------------------------------------------------------------------------------------------
with open(paths.data / f"posterior_{name}.pickle", "rb") as f:
    posterior = pickle.load(f)

# --- Load the validation data ---
# Validation data created with CHEMPY, not with the NN simulator
print("Evaluating the posterior...")
path_test = paths.data / "chempy_data/chempy_TNG_val_data.npz"
val_data = np.load(path_test, mmap_mode="r")

val_theta = val_data["params"]
val_x = val_data["abundances"]


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
abundances = torch.cat([val_x[:, :2], val_x[:, 3:]], dim=1)

# add noise to data to simulate observational errors
pc_ab = 5  # percentage error in abundance

x_err = np.ones_like(abundances) * float(pc_ab) / 100.0
abundances = norm.rvs(loc=abundances, scale=x_err)
abundances = torch.tensor(abundances).float()

theta_hat = torch.zeros_like(val_theta)
for index in tqdm(range(len(abundances))):
    thetas_predicted = posterior.sample(
        (1000,), x=abundances[index], show_progress_bars=False
    )
    theta_predicted = thetas_predicted.mean(dim=0)
    theta_hat[index] = theta_predicted

ape = torch.abs((val_theta - theta_hat) / val_theta) * 100
torch.save(ape, paths.data / f"ape_posterior_{name}.pt")

print("\nAPE of the Posterior:")
print("Median + upper quantile - lower quantile")
l_quantile, median, u_quantile = np.percentile(ape, [25, 50, 75])
print(f"Total : {median:.1f}% + {u_quantile-median:.1f} - {median-l_quantile:.1f}")
print("")

with open(paths.output / f"global_posterior_APE.txt", "w") as f:
    f.write(
        f"${median:.1f}^{{+{u_quantile-median:.1f}}}_{{-{median-l_quantile:.1f}}}\,\%$%"
    )

for i in range(ape.shape[1]):
    l_quantile, median, u_quantile = np.percentile(ape[:, i], [25, 50, 75])
    print(
        labels_in[i]
        + f" : {median:.1f}% + {u_quantile-median:.1f} - {median-l_quantile:.1f}"
    )
    if i in [0, 1]:
        with open(paths.output / f"posterior_APE.txt", "a") as f:
            f.write(
                f"${median:.1f}^{{+{u_quantile-median:.1f}}}_{{-{median-l_quantile:.1f}}}\,\%$%"
            )
