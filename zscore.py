import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas.plotting import scatter_matrix
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import r2_score, mean_squared_error


# import data
df = pd.read_csv("countryriskdata.csv", sep=",")
# select num cols
num_cols = df.columns[df.dtypes.apply(lambda x: np.issubdtype(x, np.number))]

# plot and calculate correlation
corr_matrix = df[num_cols].corr()
scatter_matrix(df[num_cols], figsize=(12, 8))
plt.show()

# regression model. predict corruption based on legal par
data_X, data_y = df[['Legal']], df['Corruption']
X_train, X_test, y_train, y_test = train_test_split(data_X, data_y,
                                                    test_size=0.33, random_state=42)

# standarize data
scaler = StandardScaler()
X_train_standarized, X_test_standarized = scaler.fit_transform(X_train), scaler.fit_transform(X_test)



# train
reg = LinearRegression().fit(X_train_standarized, y_train)
# cross validation
lin_scores = cross_validate(reg, X_train_standarized, y_train, cv=10)
# make predictions using training set
cor_pred = reg.predict(X_test_standarized)

# The coefficient of determination
print("R2: %.2f" % r2_score(y_test, cor_pred))
print("MSE: %.2f" % mean_squared_error(y_test, cor_pred))

# plot outputs
plt.scatter(X_test_standarized, y_test, color='black')
plt.plot(X_test_standarized, cor_pred, color='blue', linewidth=3)
plt.xlabel('Legal')
plt.ylabel('Corruption')

plt.xticks(())
plt.yticks(())

plt.show()
