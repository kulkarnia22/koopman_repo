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

# Filter out columns with NaN values
filtered_matrix = matrix[:, 600:700]

nan_cols = np.any(np.isnan(filtered_matrix), axis=0)

#filtered_matrix = filtered_matrix[:, ~nan_cols]

train = filtered_matrix[:(int(.9*len(filtered_matrix)))]

# Fit the model on the training data
kp.fit(train)
data_O = pykoop.extract_initial_conditions(train, min_samples = 101)
# Predict on the test data
test_predict = kp.predict_multistep(filtered_matrix[:1000])[len(train): ]

fig, axs = plt.subplots(1, 2, figsize=(10, 5))  # 1 row, 2 columns

# Determine the global vmin and vmax
vmin = min(filtered_matrix[len(train):].min(), test_predict.min())
vmax = max(filtered_matrix[len(train):].max(), test_predict.max())

im1 = axs[0].imshow(filtered_matrix[len(train):], cmap='viridis', vmin = vmin, vmax = vmax)
axs[0].set_title('Fiber Data Image')
axs[0].set_xlabel('Columns')
axs[0].set_ylabel('Rows')
cbar1 = fig.colorbar(im1, ax=axs[0])

im2 = axs[1].imshow(test_predict, cmap='viridis', vmin = vmin, vmax = vmax)
axs[1].set_title('Predicted Fiber Data Image')
axs[1].set_xlabel('Columns')
axs[1].set_ylabel('Rows')
cbar2 = fig.colorbar(im2, ax=axs[1])

plt.show()
