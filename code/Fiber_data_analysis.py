import numpy as np
from matplotlib import pyplot as plt
import sklearn
from sklearn.preprocessing import MaxAbsScaler, StandardScaler
import pykoop
import pandas as pd

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
            ('pl', pykoop.PolynomialLiftingFn(order=2)),
            ('ss', pykoop.SkLearnLiftingFn(StandardScaler())),
        ],
        regressor=pykoop.EdmdMeta(regressor=sklearn.linear_model.LinearRegression()),
    )

kp.fit(temp_data)
data_O = pykoop.extract_initial_conditions(temp_data, min_samples = 20)
data_predict = kp.predict_trajectory(data_O)
kp.plot_predicted_trajectory(temp_data)
plt.show()



