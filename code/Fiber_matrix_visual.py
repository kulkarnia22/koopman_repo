import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MaxAbsScaler, StandardScaler
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error
import pykoop
import pandas as pd

# Load data
tsv_file = 'data/LDRD_Test001-03_2019-04-24_15-23-17_ch3_full.tsv'
data_frame = pd.read_csv(tsv_file, sep='\t', header=None, skiprows=28, nrows=1000)
temp_data = data_frame.to_numpy()
data = np.array([lst[1:] for lst in temp_data])
matrix = data.astype(float)
# Plotting the matrix as an image
"""plt.imshow(matrix, cmap='viridis', interpolation='nearest')
plt.colorbar()  # Add color bar to show scale
plt.title('Fiber Data Image')
plt.xlabel('Columns')
plt.ylabel('Rows')
plt.show()"""

kp = pykoop.KoopmanPipeline(
        lifting_functions=[
            ('pl', pykoop.SkLearnLiftingFn(MaxAbsScaler())),
            ('dl', pykoop.DelayLiftingFn(n_delays_state=100)),
            ('ss', pykoop.SkLearnLiftingFn(StandardScaler())),
        ],
        #regressor=pykoop.EdmdMeta(regressor = Lasso(alpha=1e-9)),
        regressor = pykoop.Edmd(alpha=1)
        #regressor=pykoop.EdmdMeta(regressor=ElasticNet(alpha=1e-9, l1_ratio=0.5))
    )

 # Fit the model on the training data

train = matrix[:int(.9*len(matrix))]
print(len(matrix), len(train))
print(train.dtype)
print(np.isnan(matrix).any())
nan_mask = np.isnan(matrix)
np_indices = np.where(nan_mask)
print(np.where(np.isnan(train[3])))
print(train[3][1398])
#kp.fit(train)
"""data_O = pykoop.extract_initial_conditions(train, min_samples = 101)
# Predict on the test data
test_predict = kp.predict_multistep(matrix[:1000])
print(test_predict)"""

#I don't know how to work with full fiber data matrix because of presence of NaN values