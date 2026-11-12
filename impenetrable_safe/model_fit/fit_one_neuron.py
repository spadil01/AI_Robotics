import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = np.array(pd.read_csv('annacode-v.tsv', sep='\t'))

X = data[:,1]
Y = data[:,2]

# Rescale x to mean 0, std 1 for training. With raw x (20 to 100), the
# gradient for m is ~60x bigger than the gradient for b, so b learns extremely
# slowly. The final m and b are converted back to the original x units below.
x_mean = X.mean()
x_std = X.std()
Xn = (X - x_mean) / x_std

learning_rate = 0.01
epochs = 3500

# Start with the bend just right of the middle of the data (in scaled units),
# so the left arm of the V falls in the ReLU's zero region.
m = 1.0
b = -1.0

losses = []  # mean squared error recorded at each epoch

for epoch in range(epochs):
    z = m * Xn + b              # neuron's weighted input
    y = z.copy()
    y[y <= 0.] = 0.             # ReLU

    loss = np.mean((Y - y) ** 2)
    losses.append(loss)

    # ReLU derivative: 1 where the neuron is on, 0 where it outputs 0.
    # Points in the zero region contribute nothing to the gradient, which is
    # what lets the neuron ignore the left arm and fit only the right arm.
    relu_grad = (z > 0).astype(float)

    dm = np.mean(2 * (Y - y) * (-1) * relu_grad * Xn)
    db = np.mean(2 * (Y - y) * (-1) * relu_grad)

    if epoch % 100 == 0:
        print(f"Epoch {epoch}: m = {m:.3f}, b = {b:.3f}, loss = {loss:.3f}")

    m -= (learning_rate * dm)
    b -= (learning_rate * db)

# Convert back to original x units: m*(x - mean)/std + b = M*x + B
M = m / x_std
B = b - m * x_mean / x_std

print(f"Final equation: y = ReLU({M:.3f} * x + {B:.3f})")
print(f"Bend at x = {-B / M:.2f}, final loss = {losses[-1]:.3f}")

# --- Plot 1: loss vs epoch ---
plt.figure()
plt.plot(range(1, epochs + 1), losses)
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.title('One Neuron: Loss vs Epoch')
plt.grid(True)

# --- Plot 2: data and final ReLU fit ---
x_line = np.linspace(X.min(), X.max(), 400)
y_line = M * x_line + B
y_line[y_line <= 0.] = 0.

plt.figure()
plt.scatter(X, Y, s=15, label='Data')
plt.plot(x_line, y_line, color='red', label=f'Fit: y = ReLU({M:.3f}x + {B:.3f})')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('One Neuron: Data and Final Fit')
plt.legend()
plt.grid(True)

plt.show()
