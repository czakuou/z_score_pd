import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas.plotting import scatter_matrix
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# import data
df = pd.read_csv("countryriskdata.csv", sep=",")
# select num cols
num_cols = df.columns[df.dtypes.apply(lambda x: np.issubdtype(x, np.number))]

# standarize data
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# plot and calculate correlation
corr_matrix = df[num_cols].corr()
scatter_matrix(df[num_cols], figsize=(12, 8))
# plt.show()

# regression model. predict corruption based od legal par
data_X, data_y = df['Legal'], df['Corruption']
X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=0.33, random_state=42)
print(y_train)
