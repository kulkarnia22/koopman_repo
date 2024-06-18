import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MaxAbsScaler, StandardScaler
import pykoop
import pandas as pd
import openpyxl

excel_file = 'data/Ga_Test_001_2019_08_1508_15_19.xlsm'
sheet_name = 'Processed data'
data_frame = pd.read_excel(excel_file, sheet_name, usecols='F, H, K')
array = data_frame.to_numpy()
data = array

kp = pykoop.KoopmanPipeline(
        lifting_functions=[
            ('ma', pykoop.SkLearnLiftingFn(MaxAbsScaler())),
            ('pl', pykoop.PolynomialLiftingFn(order=2)),
            ('ss', pykoop.SkLearnLiftingFn(StandardScaler())),
        ],
        regressor=pykoop.Edmd(alpha=1),
    )

kp.fit(data)

data_O = pykoop.extract_initial_conditions(data, min_samples = 20)
data_predict = kp.predict_trajectory(data_O)

kp.plot_predicted_trajectory(data)
plt.show()

X = pykoop.example_data_msd()
#print(X)

#print(data)

#print(data_predict)
