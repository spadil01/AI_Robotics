import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = np.array(pd.read_csv('annacode-v.tsv', sep='\t'))

X = data[:,1]
Y = data[:,2]

# Rescale x to mean 0, std 1 for training (same reason as fit_one_neuron.py).
x_mean = X.mean()
x_std = X.std()
Xn = (X - x_mean) / x_std

learning_rate = 0.01
epochs = 3500

# Model: y = ReLU(m1*x + b1) + ReLU(m2*x + b2)
# Neuron 1 starts with a negative slope (left arm of the V),
# neuron 2 starts with a positive slope (right arm of the V).
m1, b1 = -1.0, 0.0
m2, b2 = 1.0, 0.0

losses = []  # mean squared error recorded at each epoch

for epoch in range(epochs):
    z1 = m1 * Xn + b1
    z2 = m2 * Xn + b2

    h1 = z1.copy()
    h1[h1 <= 0.] = 0.           # ReLU, neuron 1
    h2 = z2.copy()
    h2[h2 <= 0.] = 0.           # ReLU, neuron 2

    y = h1 + h2                 # output is the sum of both neurons

    loss = np.mean((Y - y) ** 2)
    losses.append(loss)

    # ReLU derivatives: 1 where each neuron is on, 0 where it outputs 0
    relu_grad1 = (z1 > 0).astype(float)
    relu_grad2 = (z2 > 0).astype(float)

    # dLoss/dy is the same for both neurons; each neuron's gradient only
    # comes from the points where that neuron is on.
    dy = 2 * (Y - y) * (-1)

    dm1 = np.mean(dy * relu_grad1 * Xn)
    db1 = np.mean(dy * relu_grad1)
    dm2 = np.mean(dy * relu_grad2 * Xn)
    db2 = np.mean(dy * relu_grad2)

    if epoch % 100 == 0:
        print(f"Epoch {epoch}: m1 = {m1:.3f}, b1 = {b1:.3f}, "
              f"m2 = {m2:.3f}, b2 = {b2:.3f}, loss = {loss:.3f}")

    m1 -= (learning_rate * dm1)
    b1 -= (learning_rate * db1)
    m2 -= (learning_rate * dm2)
    b2 -= (learning_rate * db2)

# Convert back to original x units: m*(x - mean)/std + b = M*x + B
M1 = m1 / x_std
B1 = b1 - m1 * x_mean / x_std
M2 = m2 / x_std
B2 = b2 - m2 * x_mean / x_std

print(f"Final equation: y = ReLU({M1:.3f} * x + {B1:.3f}) + ReLU({M2:.3f} * x + {B2:.3f})")
print(f"Neuron 1 bend at x = {-B1 / M1:.2f}, neuron 2 bend at x = {-B2 / M2:.2f}")
print(f"Final loss = {losses[-1]:.3f}")

# --- Plot 1: loss vs epoch ---
plt.figure()
plt.plot(range(1, epochs + 1), losses)
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.yscale('log')
plt.title('Two Neurons: Loss vs Epoch')
plt.grid(True)

# --- Plot 2: data and the final fit ---
x_line = np.linspace(X.min(), X.max(), 400)
n1_line = M1 * x_line + B1
n1_line[n1_line <= 0.] = 0.
n2_line = M2 * x_line + B2
n2_line[n2_line <= 0.] = 0.

plt.figure()
plt.scatter(X, Y, s=15, label='Data')
plt.plot(x_line, n1_line + n2_line, color='red',
         label=f'Fit: y = ReLU({M1:.3f}x + {B1:.3f}) + ReLU({M2:.3f}x + {B2:.3f})')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Two Neurons: Data and Final Fit')
plt.legend(loc='upper center', fontsize='small')
plt.grid(True)

plt.show()
