import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MaxAbsScaler, StandardScaler
import pykoop
import pandas as pd
import openpyxl

excel_file = 'data/Ga_Test_001_2019_08_1508_15_19.xlsm'
sheet_name = 'Processed data'
data_frame = pd.read_excel(excel_file, sheet_name, usecols='F, H, K')
data = data_frame.to_numpy()

kp = pykoop.KoopmanPipeline(
        lifting_functions=[
            ('ma', pykoop.SkLearnLiftingFn(MaxAbsScaler())),
            ('dl', pykoop.DelayLiftingFn(n_delays_state = 1000)),
            ('ss', pykoop.SkLearnLiftingFn(StandardScaler())),
        ],
        regressor=pykoop.Edmd(alpha=1),
    )

kp.fit(data)

data_O = pykoop.extract_initial_conditions(data, min_samples = 1001)
data_predict = kp.predict_trajectory(data_O)

print(len(data_predict))
print(len(data_O))
print(len(data[:1001]))
print(data_predict - data[:1001])
print(data_predict - data_O)

"""kp.plot_predicted_trajectory(data)
plt.show()"""

#Koopman works for this data if we use delay based lifting functions. But, it requires a much greater delay
