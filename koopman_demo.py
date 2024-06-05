import pykoop
import pydmd
from pydmd import DMD
from pydmd.plotter import plot_eigs
import pandas as pd
import openpyxl
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

excel_file = 'data/Ga_Test_001_2019_08_1508_15_19.xlsm'
sheet_name = 'Processed data'
data_frame = pd.read_excel(excel_file, sheet_name, usecols=['Outlet flow rate (liter/sec)'])
array = data_frame.to_numpy()
print(array.shape)
foo = []
for i in range(array.shape[0]):
    foo.append(array[i][0])

domain = np.arange(0,len(array), 1)

dmd = DMD(svd_rank = 2)
dmd.fit(foo.T)

for eig in dmd.eigs:
    print(
        "Eigenvalue {}: distance from unit circle {}".format(
            eig, np.abs(np.sqrt(eig.imag**2 + eig.real**2) - 1)
        )
    )

plot_eigs(dmd, show_axes=True, show_unit_circle=True)

"""for mode in dmd.modes.T:
    plt.plot(array, mode.real)
    plt.title("Modes")
plt.show()

for dynamic in dmd.dynamics:
    plt.plot(domain, dynamic.real)
    plt.title("Dynamics")
plt.show()"""



