import pykoop
import pydmd
from pydmd import DMD
from pydmd.plotter import plot_summary
from pydmd.preprocessing import hankel_preprocessing
from pydmd.plotter import plot_eigs
import pandas as pd
import openpyxl
import numpy as np
import matplotlib.pyplot as plt
import warnings
from scipy.linalg import svd
import datetime
warnings.filterwarnings('ignore')

excel_file = 'data/Ga_Test_001_2019_08_1508_15_19.xlsm'
sheet_name = 'Processed data'
data_frame = pd.read_excel(excel_file, sheet_name, usecols = 'D:R')
time_frame = pd.read_excel(excel_file, sheet_name, usecols = 'B')
time = time_frame.to_numpy().T
data = data_frame.to_numpy().T

X = data[0:len(data) - 1]

X_prime = data[1:]

U, singular, V_transpose = svd(X)

dmd = DMD(svd_rank = 2)
dmd.fit(data)

modes = dmd.modes

eigen_values = dmd.eigs

reconstructed_data = dmd.reconstructed_data

def time_to_float(t):
    total_seconds = t.hour * 3600 + t.minute * 60 + t.second + t.microsecond / 1e6
    return total_seconds / 86400


# Convert the list to floats
float_list = [time_to_float(t) for t in time[0]]


"""for i, lst in enumerate(data):
    plt.plot(float_list, lst)
    plt.plot(float_list, reconstructed_data[i])"""

plt.plot(float_list, data[13])
plt.plot(float_list, reconstructed_data[13])
    

plt.show()



plot_summary(dmd)


 