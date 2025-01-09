import paths
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from Chempy.parameter import ModelParameters

import torch
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform

import time as t
from torch_chempy_model import model

# ---  You want to retrain? ---
re_train = False

# ----- Load the data ---------------------------------------------------------------------------------------------------------------------------------------------
# ---  Load in the validation data ---
path_test = paths.data / 'chempy_data/chempy_TNG_val_data.npz'
val_data = np.load(path_test, mmap_mode='r')

val_x = val_data['params']
val_y = val_data['abundances']


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

val_x, val_y     = clean_data(val_x, val_y)

# convert to torch tensors
val_x = torch.tensor(val_x, dtype=torch.float32)
val_y = torch.tensor(val_y, dtype=torch.float32)


# ----- Define the prior ------------------------------------------------------------------------------------------------------------------------------------------
a = ModelParameters()
labels = [a.to_optimize[i] for i in range(len(a.to_optimize))] + ['time']
priors = torch.tensor([[a.priors[opt][0], a.priors[opt][1]] for opt in a.to_optimize])

# ----- Train the model -------------------------------------------------------------------------------------------------------------------------------------------


optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.MSELoss()

# --- Train the neural network ---
if re_train:
    # --- Load in training data ---
    path_training = paths.data / 'chempy_data/chempy_TNG_train_data.npz'
    training_data = np.load(path_training, mmap_mode='r')

    elements = training_data['elements']
    train_x = training_data['params']
    train_y = training_data['abundances']

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
        
            # Get the batch
            x_batch = train_x[i:i+batch_size].requires_grad_(True)
            y_batch = train_y[i:i+batch_size].requires_grad_(True)
        
            # Forward pass
            y_pred = model(x_batch)

            # Compute Loss
            loss = loss_fn(y_pred, y_batch)
            train_loss.append(loss.item())
        
            # Backward pass
            loss.backward(retain_graph=True)
            optimizer.step()
        
        # Validation loss
        y_pred = model(val_x)
        val_loss = loss_fn(y_pred, val_y)
    
        train_loss = np.array(train_loss).mean()
        ep_loss.append([train_loss, val_loss.item()])
        
        end_epoch = t.time()
        epoch_time = end_epoch - start_epoch
    
        print(f'Epoch {epoch+1}/{epochs} in {round(epoch_time,1)}s, Loss: {round(train_loss,6)} | Val Loss: {round(val_loss.item(),6)}')
    print(f'Training finished | Total time: {round(end_epoch - start, 1)}s')

    # ----- Save the model --------------------------------------------------------------------------------------------------------------------------------------------
    torch.save(model.state_dict(), paths.data / 'pytorch_state_dict.pt')
    print("Model trained and saved")


    # ----- Plot the loss -----
    ep_loss = np.array(ep_loss)

    plt.plot(np.arange(epochs)+1, ep_loss[:,0], label='Training Loss')
    plt.plot(np.arange(epochs)+1, ep_loss[:,1], label='Validation Loss')
    plt.xlabel('Epoch', fontsize=15)
    plt.ylabel('MSE Loss', fontsize=15)
    plt.title('Training and Validation Loss', fontsize=20)
    plt.legend()
    plt.tight_layout()
    plt.savefig(paths.figures / "loss_NN_simulator.png")
    plt.clf()

else:
    model.load_state_dict(torch.load(paths.data / 'pytorch_state_dict.pt'))
    print("Model loaded")

    # ----- Calculate the Absolute Percantage Error -----

    ape = 100 * torch.abs((val_y - model(val_x)) / val_y).detach().numpy()

    fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.20, .80)})

    ax_hist.hist(ape.flatten(), bins=100, density=True, range=(0, 30), color='tomato')
    ax_hist.set_xlabel('Error (%)', fontsize=15)
    ax_hist.set_ylabel('Density', fontsize=15)
    ax_hist.spines['top'].set_visible(False)
    ax_hist.spines['right'].set_visible(False)
    # percentiles
    p1,p2,p3 = np.percentile(ape, [25, 50, 75])
    ax_hist.axvline(p2, color='black', linestyle='--')
    ax_hist.axvline(p1, color='black', linestyle='dotted')
    ax_hist.axvline(p3, color='black', linestyle='dotted')
    ax_hist.text(p2, 0.2, fr'${p2:.1f}^{{+{p3-p2:.1f}}}_{{-{p2-p1:.1f}}}\%$', fontsize=12, verticalalignment='top')

    ax_box.boxplot(ape.flatten(), vert=False, autorange=False, widths=0.5, patch_artist=True, showfliers=False, boxprops=dict(facecolor='tomato'), medianprops=dict(color='black'))
    ax_box.set(yticks=[])
    ax_box.spines['left'].set_visible(False)
    ax_box.spines['right'].set_visible(False)
    ax_box.spines['top'].set_visible(False)

    fig.suptitle('APE of the Neural Network', fontsize=20)
    plt.xlim(0, 30)
    fig.tight_layout()

    plt.savefig(paths.figures / "ape_NN.png")
    plt.clf()

    with open(paths.output / 'ape_NN.txt', 'w') as f:
        f.write(f'${p2:.1f}^{{+{p3-p2:.1f}}}_{{-{p2-p1:.1f}}}\,\%$%')
