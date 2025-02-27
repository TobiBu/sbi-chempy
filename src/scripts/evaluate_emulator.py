import matplotlib.pyplot as plt
import numpy as np
import torch

import paths
from chempy_torch_model import Model_Torch

# ----- Load the data ---------------------------------------------------------------------------------------------------------------------------------------------
# ---  Load in the validation data ---
path_test = paths.data / "chempy_data/chempy_TNG_val_data.npz"
val_data = np.load(path_test, mmap_mode="r", allow_pickle=True)

val_x = val_data["params"]
val_y = val_data["abundances"]


# --- Clean the data ---
# Chempy sometimes returns zeros or infinite values, which need to be removed
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


val_x, val_y = clean_data(val_x, val_y)

# convert to torch tensors
val_x = torch.tensor(val_x, dtype=torch.float32)
val_y = torch.tensor(val_y, dtype=torch.float32)

# ----- Define the model ------------------------------------------------------------------------------------------------------------------------------------------
model = Model_Torch(val_x.shape[1], val_y.shape[1])
model.load_state_dict(torch.load(paths.data / "pytorch_state_dict.pt"))
model.eval()
print("Model loaded")

# ----- Calculate the Absolute Percantage Error -----

ape = 100 * torch.abs((val_y - model(val_x)) / val_y).detach().numpy()

fig, (ax_box, ax_hist) = plt.subplots(
    2, sharex=True, gridspec_kw={"height_ratios": (0.20, 0.80)}
)

ax_hist.hist(
    ape.flatten(),
    bins=100,
    density=True,
    cumulative=True,
    range=(0, 30),
    color="tomato",
)
ax_hist.hist(ape.flatten(), bins=100, density=True, range=(0, 30), color="tomato")
ax_hist.set_xlabel("Error (%)", fontsize=15)
ax_hist.set_ylabel("CDF", fontsize=15)
ax_hist.spines["top"].set_visible(False)
ax_hist.spines["right"].set_visible(False)
# percentiles
p1, p2, p3 = np.percentile(ape, [25, 50, 75])
ax_hist.axvline(p2, color="black", linestyle="--")
ax_hist.axvline(p1, color="black", linestyle="dotted")
ax_hist.axvline(p3, color="black", linestyle="dotted")
ax_hist.text(
    p2,
    0.2,
    rf"${p2:.1f}^{{+{p3-p2:.1f}}}_{{-{p2-p1:.1f}}}\%$",
    fontsize=12,
    verticalalignment="top",
)

ax_box.boxplot(
    ape.flatten(),
    vert=False,
    autorange=False,
    widths=0.5,
    patch_artist=True,
    showfliers=False,
    boxprops=dict(facecolor="tomato"),
    medianprops=dict(color="black"),
)
ax_box.set(yticks=[])
ax_box.spines["left"].set_visible(False)
ax_box.spines["right"].set_visible(False)
ax_box.spines["top"].set_visible(False)

# fig.suptitle('APE of the Neural Network', fontsize=20)
plt.xlim(0, 30)
fig.tight_layout()

plt.savefig(paths.figures / "ape_NN.pdf")
plt.clf()

with open(paths.output / "ape_NN.txt", "w") as f:
    f.write(f"${p2:.1f}^{{+{p3-p2:.1f}}}_{{-{p2-p1:.1f}}}\,\%$%")

with open(paths.output / "ape_NN_log.txt", "w") as f:
    f.write(f"${np.log10(1+p2/100.):.1f}\,$%")
