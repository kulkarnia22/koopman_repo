import pydmd 
from pydmd import DMD
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Load loop data
excel_file = 'data/Ga_Test_001_2019_08_1508_15_19.xlsm'
sheet_name = 'Processed data'
loop_data_frame = pd.read_excel(excel_file, sheet_name, usecols='F,H,K')
loop_data = loop_data_frame.to_numpy().T

#Load Fiber Data
tsv_file = 'data/LDRD_Test001-03_2019-04-24_15-23-17_ch3_full.tsv'
fiber_data_frame = pd.read_csv(tsv_file, sep='\t', header=None, skiprows=28, nrows=1000)
fiber_data = fiber_data_frame.to_numpy()
fiber_data = np.array([lst[1:] for lst in fiber_data], dtype = 'float64')
fiber_data = fiber_data[:, 700:800].T

#consider small input window
#First prepare the data
window_length = 50
input = loop_data[:, :window_length]
X = input[:, :-1]
X_prime = input[:, 1:]

#Conduct SVD on the input
U, S, Vh = np.linalg.svd(X, full_matrices=False)
Sigma = np.diag(S)
V = Vh.T

#Continue with DMD algorithm
#Truncate SVD values based on r
r = 20
U_r = U[:, : r]
Sigma_r = Sigma[:r, :r]
V_r = V[:, : r]

#Compute A_tilde matrix
A_tilde = U_r.T @ X_prime @ V_r @ np.linalg.inv(Sigma_r)

#Find the eigenvalues and eigenvectors of Atilde
eigenvalues, W = np.linalg.eig(A_tilde)
lamda = np.diag(eigenvalues)
w_cts = []
for i, eigenvalue in enumerate(eigenvalues):
   w_cts.append(np.log(np.abs(eigenvalue))/2)

#Use eigenvectors to find Phi
Phi = X_prime @ V_r @np.linalg.inv(Sigma_r) @ W

#Calculate the amplitudes for DMD expansion
b = np.linalg.pinv(Phi) @ X[:, 0]

#This should give me a prediction of the 10th element in my data
x_10 = 0
for i in range(Phi.shape[1]):
    x_10 = np.add(x_10, Phi[:,i]*(eigenvalues[i]**9)*b[i])

#let's try reconstructing the first 50 snapshots of the loop data
print(Phi.shape)
reconstruct_num = 50
reconstructed_matrix = []
for i in range(reconstruct_num):
  x_i = 0
  for j in range(Phi.shape[1]):
    x_i = np.add(x_i, Phi[:,j]*(eigenvalues[j]**i)*b[j])
  x_i = x_i.real
  reconstructed_matrix.append(x_i)
reconstructed_matrix = np.array(reconstructed_matrix)

#Now let's try plotting the reconstructed tc3 against the actual tc3
fig, ax = plt.subplots(figsize=(10, 10))
ax.plot(loop_data[1, :50],label='True trajectory')
ax.plot(reconstructed_matrix[:, 1], label='Reconstructed Trajectory')
plt.show()
