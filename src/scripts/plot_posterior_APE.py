import torch
from Chempy.parameter import ModelParameters

import paths
from plot_functions import ape_plot

# ----- Config -------------------------------------------------------------------------------------------------------------------------------------------
name = "NPE_C"

# --- Define the prior ---
a = ModelParameters()
labels_out = a.elements_to_trace
labels_in = [a.to_optimize[i] for i in range(len(a.to_optimize))] + ["time"]


ape = torch.load(paths.data / f"ape_posterior_{name}.pt")

save_path = paths.figures / f"ape_posterior_{name}.pdf"
ape_plot(ape, labels_in, save_path)
