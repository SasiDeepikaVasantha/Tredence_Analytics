from train import train
from utils import plot_distribution

lambdas = [1e-5, 1e-4, 1e-3]

results = []
best_model = None
best_acc = 0

for lam in lambdas:
    print(f"\nTraining with lambda = {lam}")
    model, acc, sparsity = train(lam)

    results.append((lam, acc, sparsity))

    if acc > best_acc:
        best_acc = acc
        best_model = model

print("\n===== FINAL RESULTS =====")
print("Lambda\tAccuracy\tSparsity (%)")

for r in results:
    print(f"{r[0]}\t{r[1]:.2f}\t\t{r[2]:.2f}")

# Plot best model
plot_distribution(best_model)
