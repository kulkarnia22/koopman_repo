import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MaxAbsScaler, StandardScaler
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error
import pykoop
import pandas as pd

#Load Fiber Data
tsv_file = 'data/LDRD_Test001-03_2019-04-24_15-23-17_ch3_full.tsv'
fiber_data_frame = pd.read_csv(tsv_file, sep='\t', header=None, skiprows=28, nrows=1000)
fiber_data = fiber_data_frame.to_numpy()
fiber_data = np.array([lst[1:] for lst in fiber_data])

#Load Loop Data
excel_file = 'data/Ga_Test_001_2019_08_1508_15_19.xlsm'
sheet_name = 'Processed data'
loop_data_frame = pd.read_excel(excel_file, sheet_name, usecols='F, H, K')
loop_data = loop_data_frame.to_numpy()[700:2100]

"""plt.figure(figsize=(10, 6))
plt.plot(loop_data[:, 2], label='Original Data')
plt.show()"""

kp = pykoop.KoopmanPipeline(
    lifting_functions=[
        ('pl', pykoop.SkLearnLiftingFn(MaxAbsScaler())),
        ('dl', pykoop.DelayLiftingFn(n_delays_state=1000)),
        ('ss', pykoop.SkLearnLiftingFn(StandardScaler())),
        ], regressor=pykoop.Edmd(alpha=1))

#Define training data with 9:1 ratio to test data
train = loop_data[:int(.9*len(loop_data))]

#Fit training data with pipeline object
kp.fit(train)

#Define koopman matrix
koopman_matrix = kp.regressor_.coef_

# Compute the singular values of the Koopman operator
U, S, Vt = np.linalg.svd(koopman_matrix, full_matrices=False)

min_magnitude_threshold = 1e-6
significant_singular_values = S[S >= min_magnitude_threshold]
cumulative_energy = np.cumsum(significant_singular_values**2) / np.sum(S**2)

# Determine the threshold index based on cumulative energy (e.g., 95% energy)
threshold_index = np.where(cumulative_energy >= 0.95)[0][0]

# Apply threshold to singular values
S_thresholded = np.zeros_like(S)
S_thresholded[:threshold_index] = S[:threshold_index]

# Reconstruct the Koopman operator using only significant singular values
koopman_operator_denoised = U @ np.diag(S_thresholded) @ Vt

# Update the regressor in the Koopman pipeline with the denoised Koopman operator
kp.regressor_.coef_ = koopman_operator_denoised

# Predict on the test data (or on the whole dataset to observe the denoising effect)
test_predict = kp.predict_multistep(loop_data)[len(train):]

# Plot the original and denoised predictions
plt.figure(figsize=(10, 6))
plt.plot(loop_data[:, 2][len(train):], label='Original Data')
plt.plot(test_predict[:, 2], label='Denoised Prediction')
plt.title('Original Data vs. Denoised Prediction')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.legend()
plt.show()


