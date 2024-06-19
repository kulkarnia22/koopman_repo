import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MaxAbsScaler, StandardScaler
import pykoop
import pandas as pd

tsv_file = 'data\LDRD_Test001-03_2019-04-24_15-23-17_ch3_full.tsv'
data_frame = pd.read_csv(tsv_file, sep='\t', header = None, skiprows = 28, nrows = 100)
temp_data = data_frame.to_numpy()
new = []
for lst in temp_data:
    new.append(lst[1:])
temp_data = np.array(new)


