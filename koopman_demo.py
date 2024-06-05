import pykoop
import pydmd
import pandas as pd
import openpyxl

excel_file = 'data/Ga_Test_001_2019_08_1508_15_19.xlsm'
sheet_name = 'Processed data'
data_frame = pd.read_excel(excel_file, sheet_name, usecols=['Outlet flow rate (liter/sec)'])
print(type(data_frame))

