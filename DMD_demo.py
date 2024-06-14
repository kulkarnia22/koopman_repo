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
warnings.filterwarnings('ignore')

excel_file = 'data/Ga_Test_001_2019_08_1508_15_19.xlsm'
sheet_name = 'Processed data'
data_frame = pd.read_excel(excel_file, sheet_name, usecols='D:R')
array = data_frame.to_numpy()
data = array.T

X = data[0:len(data) - 1]

X_prime = data[1:]

U, singular, V_transpose = svd(X)

dmd = DMD(svd_rank = 2)
dmd.fit(data)

modes = dmd.modes

eigen_values = dmd.eigs

reconstructed_data = dmd.reconstructed_data

plot_summary(dmd)


