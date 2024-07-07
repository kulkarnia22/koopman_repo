import pydmd 
from pydmd import DMD
import numpy as np
import pandas as pd
from scipy.linalg import svd

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
#We want the columns to represent vector states in time
window_length = 50
input = fiber_data[:, :window_length]
X = input[:, :-1]
X_prime = input[:, 1:]

#Conduct SVD on the input

U, S, Vh = np.linalg.svd(X, full_matrices=False)
Sigma = np.diag(S)
V = Vh.T

#print(U.T.shape, np.linalg.inv(Sigma).shape, V.shape, X_prime.shape)

#Truncate SVD values based on r
r = 20
U_r = U[:, : r]
Sigma_r = Sigma[:r, :r]
V_r = V[:, : r]


#Continue with rest of DMD algorithm
A_tilde = U_r.T @ X_prime @ V_r @ np.linalg.inv(Sigma_r)
A = X_prime @ V @ np.linalg.inv(Sigma) @ U.T


#Find the eigenvalues and eigenvectors of Atilde
eigenvalues, W = np.linalg.eig(A_tilde)
lamda = np.diag(eigenvalues)

eig_A, eig_vectors_A = np.linalg.eig(A)

#Use eigenvectors to find Phi
Phi = X_prime @ V_r @np.linalg.inv(Sigma_r) @ W

#Calculate the amplitudes for DMD expansion
#Might need a better method for calculating b
b = np.linalg.pinv(Phi) @ X[:, 0]

#This should give me a prediction of the 10th element in my data
x_10 = 0
for i in range(Phi.shape[1]):
    x_10 = np.add(x_10, Phi[:,i]*(eigenvalues[i]**9)*b[i])

#I think this works well for fiber data where m << n. For loop data where m >> n, I might need
#to try something else

