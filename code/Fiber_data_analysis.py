import numpy as np
from matplotlib import pyplot as plt
import sklearn
from sklearn.preprocessing import MaxAbsScaler, StandardScaler, MinMaxScaler
import pykoop
import pandas as pd
import scipy

tsv_file = 'data\LDRD_Test001-03_2019-04-24_15-23-17_ch3_full.tsv'
data_frame = pd.read_csv(tsv_file, sep='\t', header = None, skiprows = 28, nrows = 500)
temp_data = data_frame.to_numpy()
new = []
for lst in temp_data:
    new.append(lst[1:3])

temp_data = np.array(new)

kp = pykoop.KoopmanPipeline(
        lifting_functions=[
            ('ma', pykoop.SkLearnLiftingFn(MaxAbsScaler())),
            ('dl', pykoop.DelayLiftingFn(n_delays_state = 100)),
            ('ss', pykoop.SkLearnLiftingFn(StandardScaler())),
        ],
        regressor=pykoop.Edmd(alpha=1),
    )

"""kp = pykoop.KoopmanPipeline(
        lifting_functions=[(
            'sp',
            pykoop.SplitPipeline(
                lifting_functions_state=[
                    ('pl', pykoop.PolynomialLiftingFn(order=3))
                ],
                lifting_functions_input=None,
            ),
        )],
        regressor=pykoop.Edmd(),
    )
"""

"""kp = pykoop.KoopmanPipeline(
    lifting_functions=[(
        'rbf',
        pykoop.RbfLiftingFn(
        rbf='thin_plate',
        centers=pykoop.QmcCenters(
            n_centers=100,
            qmc=scipy.stats.qmc.LatinHypercube,
            random_state=666,
                ),
            ),
        )],
        regressor=pykoop.Edmd(),
)"""

kp.fit(temp_data)
data_O = pykoop.extract_initial_conditions(temp_data, min_samples = 101)
data_predict = kp.predict_trajectory(data_O)
predict = kp.predict_multistep(temp_data)


"""kp.plot_predicted_trajectory(temp_data)
plt.show()"""

print(predict - temp_data)
# Plot trajectories in phase space
fig, ax = plt.subplots(constrained_layout=True, figsize=(6, 6))
ax.plot(
    temp_data[:, 1],
    label='True trajectory'
)
ax.plot(predict[:, 1],label='Local prediction')

plt.show()

"""kp.plot_koopman_matrix()
plt.show()"""



