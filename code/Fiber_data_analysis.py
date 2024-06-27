import numpy as np
from matplotlib import pyplot as plt
import sklearn
from sklearn.preprocessing import MaxAbsScaler, StandardScaler, MinMaxScaler
from sklearn.linear_model import Lasso
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import pykoop
import pandas as pd
import scipy

tsv_file = 'data\LDRD_Test001-03_2019-04-24_15-23-17_ch3_full.tsv'
data_frame = pd.read_csv(tsv_file, sep='\t', header = None, skiprows = 28, nrows = 500)
temp_data = data_frame.to_numpy()
new = []
for lst in temp_data:
    new.append([lst[600]])

data = np.array(new)
train = data[:400]

kp = pykoop.KoopmanPipeline(
        lifting_functions=[
            ('ma', pykoop.SkLearnLiftingFn(MaxAbsScaler())),
            ('dl', pykoop.DelayLiftingFn(n_delays_state = 350)),
            ('ss', pykoop.SkLearnLiftingFn(StandardScaler())),
        ],
        #regressor=pykoop.EdmdMeta(regressor=Lasso(alpha=1e-9)),
        regressor = pykoop.Edmd(alpha=1)
    )


kp.fit(train)
data_O = pykoop.extract_initial_conditions(train, min_samples = 351)
data_predict = kp.predict_trajectory(data_O)
predict = kp.predict_multistep(data)


fig, ax = plt.subplots(figsize=(10, 10))
ax.plot(data[:, 0][400:],label='True trajectory')
ax.plot(predict[:, 0][400:],label='Local prediction')

plt.show()


