import pandas as pd # Used for data manipulation and analysis with data frames
from sklearn.linear_model import LinearRegression # Imports the Linear Regression model from scikit-learn
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle

df = pd.read_csv(io.BytesIO('SeaLevelsSince1880.csv'), header = 0)
df = df.drop(['adjlev_noaa', 'rownames'], axis=1)

#-------- Building the Linear Regression model--------------
# 'year' is the independent variable and 'adjlev' is the dependent variable
df = df.dropna(subset=['adjlev'])  # Drop rows where 'adjlev' has value of Nan
X = df[['year']]  # Reshaping to a 2D array
y = df['adjlev']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)  # spliting the data: 75% train, 25% test

model = LinearRegression()
model.fit(X_train, y_train)  # Training the model

y_pred = model.predict(X_test)  # Testing the model

with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

print("model trained and tested")


