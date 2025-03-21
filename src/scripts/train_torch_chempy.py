import time as t

import matplotlib.pyplot as plt
import numpy as np
import schedulefree
import torch
from Chempy.parameter import ModelParameters
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform

import paths
from chempy_torch_model import Model_Torch

# ---  You want to retrain? ---
re_train = True  # False

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


# ----- Train the model -------------------------------------------------------------------------------------------------------------------------------------------

# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss()


# --- Train the neural network ---
if re_train:
    # --- Load in training data ---
    path_training = (
        paths.data / "chempy_data/chempy_train_uniform_prior_5sigma.npz"
    )  # chempy_TNG_train_data.npz'
    training_data = np.load(path_training, mmap_mode="r")

    elements = training_data["elements"]
    train_x = training_data["params"]
    train_y = training_data["abundances"]

    train_x, train_y = clean_data(train_x, train_y)

    train_x = torch.tensor(train_x, dtype=torch.float32)
    train_y = torch.tensor(train_y, dtype=torch.float32)

    # shuffle the data
    index = np.arange(train_x.shape[0])
    np.random.shuffle(index)
    train_x = train_x[index]
    train_y = train_y[index]

    print("Training the model")
    epochs = 20
    batch_size = 64
    ep_loss = []
    start = t.time()
    for epoch in range(epochs):
        start_epoch = t.time()
        train_loss = []
        for i in range(0, train_x.shape[0], batch_size):
            optimizer.zero_grad()
            optimizer.train()

            # Get the batch
            x_batch = train_x[i : i + batch_size].requires_grad_(True)
            y_batch = train_y[i : i + batch_size].requires_grad_(True)

            # Forward pass
            y_pred = model(x_batch)

            # Compute Loss
            loss = loss_fn(y_pred, y_batch)
            train_loss.append(loss.item())

            # Backward pass
            loss.backward(retain_graph=True)
            optimizer.step()

        # Validation loss
        optimizer.eval()
        y_pred = model(val_x)
        val_loss = loss_fn(y_pred, val_y)

        train_loss = np.array(train_loss).mean()
        ep_loss.append([train_loss, val_loss.item()])

        end_epoch = t.time()
        epoch_time = end_epoch - start_epoch

        print(
            f"Epoch {epoch+1}/{epochs} in {round(epoch_time,1)}s, Loss: {round(train_loss,6)} | Val Loss: {round(val_loss.item(),6)}"
        )
    print(f"Training finished | Total time: {round(end_epoch - start, 1)}s")

    # ----- Save the model --------------------------------------------------------------------------------------------------------------------------------------------
    torch.save(model.state_dict(), paths.data / "pytorch_state_dict.pt")
    print("Model trained and saved")

    # ----- Plot the loss -----
    ep_loss = np.array(ep_loss)

    plt.plot(np.arange(epochs) + 1, ep_loss[:, 0], label="Training Loss")
    plt.plot(np.arange(epochs) + 1, ep_loss[:, 1], label="Validation Loss")
    plt.xlabel("Epoch", fontsize=15)
    plt.ylabel("MSE Loss", fontsize=15)
    plt.title("Training and Validation Loss", fontsize=20)
    plt.legend()
    plt.tight_layout()
    plt.savefig(paths.figures / "loss_NN_simulator.pdf")
    plt.clf()
