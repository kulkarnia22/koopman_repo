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

# Load data
tsv_file = 'data/LDRD_Test001-03_2019-04-24_15-23-17_ch3_full.tsv'
data_frame = pd.read_csv(tsv_file, sep='\t', header=None, skiprows=28, nrows=1000)
temp_data = data_frame.to_numpy()
data = np.array([lst[1] for lst in temp_data]).reshape(-1,1)

# Set up cross-validation
n_splits = 25
tscv = TimeSeriesSplit(n_splits=n_splits)
mse_scores = []
window_and_test_size = []
train_indeces = []

# Cross-validation loop
for fold, (train_index, test_index) in enumerate(tscv.split(data)):
    train, test = data[train_index], data[test_index]
    window_and_test_size.append((len(train), len(test)))
    train_indeces.append(train_index)
    num_delays = int(len(train)/2) + int(len(train)/4)
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
    test_predict = kp.predict_multistep(np.concatenate((train, test), axis = 0))[len(train):]

    # Evaluate the model
    mse = mean_squared_error(test, test_predict)
    mse_scores.append(mse)

    # Plotting
    fig, ax = plt.subplots(constrained_layout=True, figsize=(10, 6))
    ax.plot(test, label='True trajectory')
    ax.plot(test_predict, label='Predicted trajectory')
    ax.set_title(f'Fold {len(mse_scores)}')
    ax.legend()
    plt.show()

# Average MSE across all folds
print(window_and_test_size)
average_mse = np.mean(mse_scores)
print(f'Average MSE: {average_mse}')

"""train1 = data[train_indeces[0]]
train2 = data[train_indeces[1]]
train3 = data[train_indeces[2]]
train4 = data[train_indeces[3]]
train5 = data[train_indeces[4]]
num_delays = int(len(train1)/2)

kp = pykoop.KoopmanPipeline(
        lifting_functions=[
            ('pl', pykoop.SkLearnLiftingFn(MaxAbsScaler())),
            ('dl', pykoop.DelayLiftingFn(n_delays_state=num_delays)),
            ('ss', pykoop.SkLearnLiftingFn(StandardScaler())),
        ],
        regressor=pykoop.EdmdMeta(regressor = Lasso(alpha=1e-9)),
        #regressor = pykoop.Edmd(alpha=1)
    )

# Fit the model on the training data
kp.fit(train1)
data_O = pykoop.extract_initial_conditions(train1, min_samples=num_delays + 1)

# Predict on the test data
test_predict = kp.predict_multistep(data)[:len(train1)]

fig, ax = plt.subplots(constrained_layout=True, figsize=(10, 6))
ax.plot(test, label='True trajectory')
ax.plot(test_predict, label='Predicted trajectory')
#ax.set_title(f'Fold {len(mse_scores)}')
ax.legend()
plt.show()"""

