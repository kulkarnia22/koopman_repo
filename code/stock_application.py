import numpy as np
from matplotlib import pyplot as plt
import sklearn
from sklearn.preprocessing import MaxAbsScaler, StandardScaler, MinMaxScaler
import pykoop
import pandas as pd
import scipy

csv_file = 'data\AAPL.csv'
data_frame = pd.read_csv(csv_file, usecols=[1,2,3])
data = data_frame.to_numpy()

kp = pykoop.KoopmanPipeline(
        lifting_functions=[
            ('ma', pykoop.SkLearnLiftingFn(MaxAbsScaler())),
            ('dl', pykoop.DelayLiftingFn(n_delays_state = 174)),
            ('ss', pykoop.SkLearnLiftingFn(StandardScaler())),
        ],
        regressor=pykoop.Edmd(alpha=1),
    )

kp.fit(data)
data_O = pykoop.extract_initial_conditions(data, min_samples = 175)
data_predict = kp.predict_trajectory(data_O)
predict = kp.predict_multistep(data)

print(kp.min_samples_)
# Plot trajectories in phase space
fig, ax = plt.subplots(constrained_layout=True, figsize=(6, 6))
ax.plot(data[:, 1],label='True trajectory')
ax.plot(predict[:, 1],label='Local prediction')

plt.show()