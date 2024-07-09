import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load loop data
excel_file = 'data/Ga_Test_001_2019_08_1508_15_19.xlsm'
sheet_name = 'Processed data'
loop_data_frame = pd.read_excel(excel_file, sheet_name, usecols='F,H,K')
loop_data = loop_data_frame.to_numpy().T

# Load Fiber Data
tsv_file = 'data/LDRD_Test001-03_2019-04-24_15-23-17_ch3_full.tsv'
fiber_data_frame = pd.read_csv(tsv_file, sep='\t', header=None, skiprows=28, nrows=1000)
fiber_data = fiber_data_frame.to_numpy()
fiber_data = np.array([lst[1:] for lst in fiber_data], dtype='float64')
fiber_data = fiber_data[:, 700:800].T

# Consider small input window
window_length = 50
input_data = loop_data[:, :window_length]
X = input_data[:, :-1]
X_prime = input_data[:, 1:]

# Parameters for residual DMD
r = 20
max_iterations = 5
tolerance = 1e-6

# Function to perform standard DMD
def perform_dmd(X, X_prime, r):
    U, S, Vh = np.linalg.svd(X, full_matrices=False)
    Sigma = np.diag(S)
    V = Vh.T

    U_r = U[:, :r]
    Sigma_r = Sigma[:r, :r]
    V_r = V[:, :r]

    A_tilde = U_r.T @ X_prime @ V_r @ np.linalg.inv(Sigma_r)
    eigenvalues, W = np.linalg.eig(A_tilde)
    Phi = X_prime @ V_r @ np.linalg.inv(Sigma_r) @ W
    b = np.linalg.pinv(Phi) @ X[:, 0]
    
    return Phi, eigenvalues, b

# Iteratively apply residual DMD
reconstructed_matrix = np.zeros_like(X_prime)
residual = X_prime

for iteration in range(max_iterations):
    Phi, eigenvalues, b = perform_dmd(X, residual, r)
    
    # Reconstruct the data using DMD modes
    for i in range(X_prime.shape[1]):
        x_i = 0
        for j in range(Phi.shape[1]):
            x_i = np.add(x_i, Phi[:, j] * (eigenvalues[j] ** i) * b[j])
        x_i = x_i.real
        reconstructed_matrix[:, i] = x_i

    # Calculate residual
    new_residual = X_prime - reconstructed_matrix
    if np.linalg.norm(new_residual) < tolerance:
        break
    residual = new_residual

# Plot the results
fig, ax = plt.subplots(figsize=(10, 10))
ax.plot(loop_data[1, :50], label='True trajectory')
ax.plot(reconstructed_matrix[1, :], label='Reconstructed Trajectory')
plt.legend()
plt.show()
