# Driving-Business-Growth-through-Sales-Prediction
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
data=pd.read_csv(r"C:\Users\diksh\Desktop\advertising (1).csv")
data.head()
data.shape
data.dtypes
data.isna().sum()
data.describe()
#Separate features (X) and target (y)
X = data[['TV', 'Radio', 'Newspaper']]
y = data['Sales']
#Model Training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#Creating a Random Forest regression model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)
#Model Evaluation
y_pred = model.predict(X_test)
#Evaluating the model's performance
mse = mean_squared_error(y_test, y_pred)
r_squared = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("R-squared:", r_squared)
