import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MaxAbsScaler, StandardScaler
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error
import pykoop
import pandas as pd


"""
What time series cross validation does is split the data into a training and test group.
The intial trainiing and test group only combine to form a small part of the data. With each
iteration in the for loop, the past test data is combined with the past training data to form
the new training data while adding more data from the actual data to form the new test data. 
This process continues until all the data is covered. 
"""

# Load fiber data
tsv_file = 'data/LDRD_Test001-03_2019-04-24_15-23-17_ch3_full.tsv'
data_frame = pd.read_csv(tsv_file, sep='\t', header=None, skiprows=28, nrows=1000)
temp_data = data_frame.to_numpy()
print(temp_data.shape)
data = np.array([lst[1200] for lst in temp_data]).reshape(-1,1)

#Load water loop data
"""excel_file = 'data/Ga_Test_001_2019_08_1508_15_19.xlsm'
sheet_name = 'Processed data'
data_frame = pd.read_excel(excel_file, sheet_name, usecols='F, H, K')
data = data_frame.to_numpy()"""

"""fig, ax = plt.subplots(constrained_layout=True, figsize=(10, 6))
ax.plot(data[:,0], label='True trajectory')
plt.show()"""

# Set up cross-validation
n_splits = 20
tscv = TimeSeriesSplit(n_splits=n_splits)
mse_scores = []
window_and_test_size = []
train_indeces = []

# Cross-validation loop
for fold, (train_index, test_index) in enumerate(tscv.split(data)):
    train, test = data[train_index], data[test_index]
    window_and_test_size.append((len(train), len(test)))
    train_indeces.append(train_index)
    num_delays = int(.66*len(train))
    # Initialize the Koopman pipeline
    kp = pykoop.KoopmanPipeline(
        lifting_functions=[
            ('pl', pykoop.SkLearnLiftingFn(MaxAbsScaler())),
            ('dl', pykoop.DelayLiftingFn(n_delays_state=num_delays)),
            ('ss', pykoop.SkLearnLiftingFn(StandardScaler())),
        ],
        #regressor=pykoop.EdmdMeta(regressor = Lasso(alpha=1e-9)),
        regressor = pykoop.Edmd(alpha=1)
        #regressor=pykoop.EdmdMeta(regressor=ElasticNet(alpha=1e-9, l1_ratio=0.5))
    )

    # Fit the model on the training data
    data_O = pykoop.extract_initial_conditions(train, min_samples = num_delays + 1)
    kp.fit(train)
    # Predict on the test data
    #Define koopman matrix
    koopman_matrix = kp.regressor_.coef_

    # Compute the singular values of the Koopman operator
    U, S, Vt = np.linalg.svd(koopman_matrix, full_matrices=False)

    min_magnitude_threshold = 1e-6
    significant_singular_values = S[S >= min_magnitude_threshold]
    cumulative_energy = np.cumsum(significant_singular_values**2) / np.sum(S**2)

    # Determine the threshold index based on cumulative energy (e.g., 95% energy)
    threshold_index = np.where(cumulative_energy >= 0.98)[0][0]

    # Apply threshold to singular values
    S_thresholded = np.zeros_like(S)
    S_thresholded[:threshold_index] = S[:threshold_index]

    # Reconstruct the Koopman operator using only significant singular values
    koopman_operator_denoised = U @ np.diag(S_thresholded) @ Vt

    # Update the regressor in the Koopman pipeline with the denoised Koopman operator
    kp.regressor_.coef_ = koopman_operator_denoised
    test_predict = kp.predict_multistep(np.concatenate((train, test), axis = 0))[len(train):]

    # Evaluate the model
    mse = mean_squared_error(test, test_predict)
    mse_scores.append(mse)

    # Plotting
    fig, ax = plt.subplots(constrained_layout=True, figsize=(10, 10))
    ax.plot(test[:,0], label='True trajectory')
    ax.plot(test_predict[:,0], label='Predicted trajectory')
    ax.set_title(f'Fold {len(mse_scores)}')
    ax.legend()
    plt.show()

# Average MSE across all folds
print(window_and_test_size)
average_mse = np.mean(mse_scores)
print(f'Average MSE: {average_mse}')

