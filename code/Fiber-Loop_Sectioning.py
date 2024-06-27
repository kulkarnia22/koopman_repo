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
fiber_data = fiber_data[:, 700:800]

#Load Loop Data
excel_file = 'data/Ga_Test_001_2019_08_1508_15_19.xlsm'
sheet_name = 'Processed data'
loop_data_frame = pd.read_excel(excel_file, sheet_name, usecols='F, H, K')
loop_data = loop_data_frame.to_numpy()

"""
Structure:

Define a function that splits the data into multiple chunks.

For each chunk, define a function that separates the chunk into a section for training and
a section for testing.

In that function, create a koopman pipeline object and fit it to the training data. 

Plot the prediction against the test and return the koopman matrix.

Experiment with singular value decomposition for noise reduction. Consider creating a 
separate function for this. 

"""

def split(data, num_splits):
    length = len(data)
    split_index = int(length/num_splits)
    splits = []
    j = 0
    while j + split_index < length:
        splits.append(data[j:j+split_index])
        j+=split_index
    return splits 

def predict(chunk, make_plot = True):
    chunk = np.array(chunk)
    train = chunk[:int(.9*len(chunk))]
    kp = pykoop.KoopmanPipeline(
    lifting_functions=[
        ('pl', pykoop.SkLearnLiftingFn(MaxAbsScaler())),
        ('dl', pykoop.DelayLiftingFn(n_delays_state=int(.66*len(train)))),
        ('ss', pykoop.SkLearnLiftingFn(StandardScaler())),
        ], regressor=pykoop.Edmd(alpha=1))
    
    kp.fit(train)
    koopman_matrix = kp.regressor_.coef_
    # Predict on the test data (or on the whole dataset to observe the denoising effect)
    test_predict = kp.predict_multistep(chunk)[len(train):]

    if make_plot:
        plt.figure(figsize=(10, 6))
        plt.plot(chunk[:, 50][len(train):], label='Original Data')
        plt.plot(test_predict[:, 50], label='Prediction')
        plt.title('Original Data vs Prediction')
        plt.xlabel('Time Step')
        plt.ylabel('Value')
        plt.legend()
        plt.show()

    return koopman_matrix

def find_average_koopman(chunks):
    koopmans = []
    for chunk in chunks:
        predicted = predict(chunk, False)
        koopmans.append(predicted)
    koopmans = np.array(koopmans)
    stacked = np.stack(koopmans, axis=0)
    average = np.mean(stacked, axis=0)
    return average


chunks = split(loop_data, 5)
whole = np.vstack(chunks)
train = chunks[3]
num_delays = int(.66*len(train))
kp_actual = pykoop.KoopmanPipeline(
    lifting_functions=[
        ('pl', pykoop.SkLearnLiftingFn(MaxAbsScaler())),
        ('dl', pykoop.DelayLiftingFn(n_delays_state=num_delays)),
        ('ss', pykoop.SkLearnLiftingFn(StandardScaler())),
        ], regressor=pykoop.Edmd(alpha=1))

kp_actual.fit(train)
print(kp_actual.regressor_.coef_.shape)
print(" ")
kp_actual.regressor_.coef_ = find_average_koopman(chunks[:len(chunks) - 1])
print(" ")
print(kp_actual.regressor_.coef_.shape)
data_O = pykoop.extract_initial_conditions(train, min_samples = num_delays + 1)
predict_data = kp_actual.predict_multistep(whole)[len(whole) - len(train):]
print(predict_data)


"""fig, ax = plt.subplots(figsize=(10, 10))
ax.plot(whole[:, 0],label='True trajectory')
ax.plot(predict_data[:, 0],label='Local prediction')

plt.show()"""

"""
I want to split the data into chunks. I want to conduct koopman analysis on each chunk except the last.
I want to average the koopman operators I've found for each chunk and use the average to predict the 
last chunk. Then I want to plot the actual last chunk against the predicted.
"""


    

