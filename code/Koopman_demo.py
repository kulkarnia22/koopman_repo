import numpy as np
from matplotlib import pyplot as plt
import pykoop
import pandas as pd
import openpyxl


excel_file = 'data/Ga_Test_001_2019_08_1508_15_19.xlsm'
sheet_name = 'Processed data'
data_frame = pd.read_excel(excel_file, sheet_name, usecols='D:R')
array = data_frame.to_numpy()
data = array.T

# Define the DMD regressor
dmd = pykoop.Dmd()

# Create a KoopmanPipeline object with the DMD regressor
koopman_pipeline = pykoop.KoopmanPipeline(regressor = dmd)

# Fit the Koopman model to the data
koopman_pipeline.fit(data)

# Get the Koopman eigenvalues and eigenvectors
koopman_eigenvalues = koopman_pipeline.regressor_.eigenvalues_
koopman_modes = koopman_pipeline.regressor_.modes_

print("Koopman Eigenvectors:\n", koopman_modes)
print("Koopman Eigenvalues:\n", koopman_eigenvalues)

plt.figure(figsize=(10, 6))
for i in range(koopman_modes.shape[1]):
    plt.plot(koopman_modes[:, i], label=f"Mode {i+1}")

plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Koopman Modes (Eigenfunctions)')
plt.legend()
plt.grid(True)
plt.show()

"""# Reconstruction function
def reconstruct_data(modes, eigenvalues, timestep):
    # Number of modes
    k = modes.shape[1]
    
    # Time vector (for illustration)
    time_vector = np.arange(data.shape[1])  # Assuming time vector matches data points
    
    # Initialize reconstructed data with complex dtype
    reconstructed_data = np.zeros_like(data, dtype=np.complex128)
    
    # Iterate over modes and reconstruct data
    for i in range(k):
        mode_i = modes[:, i:i+1]  # Take each mode as a column vector
        eigenvalue_i = eigenvalues[i]
        
        # Compute contribution of each mode
        mode_contribution = np.dot(mode_i, np.exp(eigenvalue_i * time_vector[timestep]))
        
        # Accumulate contributions to reconstructed data
        reconstructed_data += mode_contribution
    
    # Convert reconstructed_data to float64
    reconstructed_data = np.real(reconstructed_data)
    
    return reconstructed_data

# Example: Reconstruct data at a specific timestep (e.g., timestep 0)
reconstructed_data = reconstruct_data(koopman_modes, koopman_eigenvalues, timestep=0)

# Print or use reconstructed_data as needed
print("Reconstructed Data at timestep 0:\n", reconstructed_data)"""

