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
train = data[:2200]

kp = pykoop.KoopmanPipeline(
        lifting_functions=[
            ('ma', pykoop.SkLearnLiftingFn(MaxAbsScaler())),
            ('dl', pykoop.DelayLiftingFn(n_delays_state = 1200)),
            ('ss', pykoop.SkLearnLiftingFn(StandardScaler())),
        ],
        regressor=pykoop.Edmd(alpha=1),
    )


data_O = pykoop.extract_initial_conditions(data, min_samples = 1201)
kp.fit(train)
predict = kp.predict_multistep(data)[len(train):]

fig, ax = plt.subplots(constrained_layout=True, figsize=(10, 10))
ax.plot(data[len(train):][:,0], label='True trajectory')
ax.plot(predict[:,0], label='Predicted trajectory')
ax.legend()
plt.show()

