import pydmd 
from pydmd import DMD
import numpy as np
import pandas as pd

#Load loop data
excel_file = 'data/Ga_Test_001_2019_08_1508_15_19.xlsm'
sheet_name = 'Processed data'
loop_data_frame = pd.read_excel(excel_file, sheet_name, usecols='F,H,K')
loop_data = loop_data_frame.to_numpy()

#Load Fiber Data
tsv_file = 'data/LDRD_Test001-03_2019-04-24_15-23-17_ch3_full.tsv'
fiber_data_frame = pd.read_csv(tsv_file, sep='\t', header=None, skiprows=28, nrows=1000)
fiber_data = fiber_data_frame.to_numpy()
fiber_data = np.array([lst[1:] for lst in fiber_data], dtype = 'float64')
fiber_data = fiber_data[:, 700:800]

#consider small input window
#Might have to work with input.T instead
window_length = 50
input = fiber_data[:window_length].T
X = input[:, :-1]
X_prime = input[:, 1:]

#Conduct SVD on the input
U, S, V = np.linalg.svd(X, full_matrices=False)
Sigma = np.diag(S)


#Truncate SVD values based on r
r = 5
U_r = U[:, : r]
Sigma_r = Sigma[:r, :r]
V_r = V[:, : r]

"""print(np.linalg.inv(Sigma_r).shape)
print(V_r.shape)
print(X_prime.shape)
print(U_r.T.shape)"""
#Continue with rest of DMD algorithm
A_tilde = U_r.T @ X_prime @ V_r @ np.linalg.inv(Sigma_r)

#Find the eigenvalues and eigenvectors of Atilde
eigenvalues, eigenvectors = np.linalg.eig(A_tilde)

#Use eigenvectors to find Phi
Phi = X_prime @ V_r @np.linalg.inv(Sigma_r) @ eigenvectors

#Calculate the amplitudes for DMD expansion
#Might need a better method for calculating b
b = np.linalg.pinv(Phi) @ X[:, 0]

#I am currently using the first 50 snapshots. What if I want to predict the next 
#10 snapshots?
print("THIS IS b ")
print(" ")
print(np.diag(b).shape)
print(" ")
print("THESE ARE THE EIGENVALUES ")
print(" ")
print(np.diag(eigenvalues).shape)
print(" ")
print((Phi @ np.linalg.multi_dot([np.diag(b), np.power(np.diag(eigenvalues),10)])).real)
print(X.shape)
