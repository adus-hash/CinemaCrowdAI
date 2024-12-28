from math import floor
import pandas
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import OrdinalEncoder


model_param = ["Rating", "Release Year"]

# NACITANIE DATASETU
df = pandas.read_csv("kino_dataset.csv")
data = df[model_param]
target = df["Attendance (%)"]


# ENCODER - potrebujem zmenit data typu string na ciselnu
encoder = OrdinalEncoder()
df[["Month of the Year", "Genre"]] = encoder.fit_transform(df[["Month of the Year", "Genre"]])

# PREPROCESSING - vypln 0 vsade kde chyba hodnota
df.fillna(0, inplace=True)

indx_filmu = 1  # pre skusku z datasetu
new_films = pandas.DataFrame([[8, 2015], [5, 1999], [2, 2023], [9, 1974]], columns=model_param)

# Covariance - meria ako sa ovplyvnuju dve premenne
"""
output:
    > 1 - linearny vztah premennych
    < 1 - zaporny linearny vztah
    ~ 0 - nelinearny vztah
"""
covariance = df.cov()
print("Covariance:\n", covariance["Attendance (%)"].to_string(), "\n")


# ZISTENIE KORELACII
corelation = df.corr()
target_corr = corelation["Attendance (%)"]

print("Correlation:\n", target_corr.sort_values(ascending=False), "\n")
print("Data info:\n", data.describe(), "\n")

# Support Vector Machine - _svm
model_svm = SVR()
model_svm.fit(data, target)

predicted_svm = model_svm.predict(data)
mse_svm = floor(mean_squared_error(target, predicted_svm))
mae_svm = floor(mean_absolute_error(target, predicted_svm))


# Random Forest Regression - _rf
model_rf = RandomForestRegressor()
model_rf.fit(data, target)

predicted_rf = model_rf.predict(data)
mse_rf = floor(mean_squared_error(target, predicted_rf))
mae_rf = floor(mean_absolute_error(target, predicted_rf))


# Decision Tree - _dt
model_dt = DecisionTreeRegressor()
model_dt.fit(data, target)

predicted_dt = model_dt.predict(data)
mse_dt = floor(mean_squared_error(target, predicted_dt))
mae_dt = floor(mean_absolute_error(target, predicted_dt))


# Linear Regression - _li
model_li = LinearRegression()
model_li.fit(data, target)

predicted_li = model_li.predict(data)
mse_li = floor(mean_squared_error(target, predicted_li))
mae_li = floor(mean_absolute_error(target, predicted_li))

print("Linear Regression, Decission Tree, Random Forest, SVM")
print(mse_li, mse_dt, mse_rf, mse_svm)
print(mae_li, mae_dt, mae_rf, mae_svm)
print(predicted_li[indx_filmu], predicted_dt[indx_filmu], predicted_rf[indx_filmu], predicted_svm[indx_filmu], df["Attendance (%)"][indx_filmu])
print(model_li.predict(new_films), model_dt.predict(new_films), model_rf.predict(new_films), model_svm.predict(new_films))

# na 3D projekciu
fig = plt.figure()
graf = fig.add_subplot(projection='3d')

graf.scatter(data["Release Year"], data["Rating"], target)
graf.set_xlabel("year")
graf.set_ylabel("rating")
graf.set_zlabel("attendance")

plt.show()
