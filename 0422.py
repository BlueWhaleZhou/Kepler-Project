import numpy as np
import pandas as pd

file_name = "/home/qinghai/Zhou/Data.csv"
data = pd.read_csv(file_name, header=0, delimiter=',')
print data
data = data.drop_duplicates()
data.drop(data.index[17], axis=0)
def change(x):
	if type(x) == str:
		if not x.lstrip("-").isdigit():
			return np.nan
	return x
gender = data['Gender']
sid = data['Student ID']
data = data.applymap(change)
data['Gender'] = gender
data['Student ID'] = sid
print data
