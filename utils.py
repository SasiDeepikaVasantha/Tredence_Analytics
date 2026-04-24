import torch
import numpy as np
import matplotlib.pyplot as plt

# =========================
# SPARSITY LOSS
# =========================
def sparsity_loss(model):
    loss = 0
    for layer in [model.fc1, model.fc2, model.fc3]:
        gates = layer.get_gates()
        loss += torch.sum(gates)
    return loss


# =========================
# CALCULATE SPARSITY
# =========================
def calculate_sparsity(model, threshold=1e-2):
    gates = model.get_all_gates().detach().cpu().numpy()
    total = len(gates)
    pruned = np.sum(gates < threshold)
    return 100 * pruned / total


# =========================
# PLOT DISTRIBUTION
# =========================
def plot_distribution(model):
    gates = model.get_all_gates().detach().cpu().numpy()

    plt.hist(gates, bins=50)
    plt.title("Gate Value Distribution")
    plt.xlabel("Gate Value")
    plt.ylabel("Frequency")
    plt.show()
