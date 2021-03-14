import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from pandas.plotting import scatter_matrix

# import data
df = pd.read_csv("countryriskdata.csv", sep=",")
# select num cols
num_cols = df.columns[df.dtypes.apply(lambda x: np.issubdtype(x, np.number))]

# standarize data
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# plot and calculate correlation
corr_matrix = df[num_cols].corr()
print(corr_matrix)
scatter_matrix(df[num_cols], figsize=(12, 8))
plt.show()
