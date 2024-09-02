import numpy as np
import pandas as pd
import tensorflow as tf
# Kai chun 
## Data Preprocessing
# Importing the dataset
dataset = pd.read_excel('Two_Axle_Sim_Data.xlsx')
X = dataset.iloc[2:, :13].values
Y = dataset.iloc[2:, 22].values


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

X_train = np.array(X_train, dtype=np.float32)
Y_train = np.array(Y_train, dtype=np.float32)
print("X_train max:", np.max(X_train))
print("X_train min:", np.min(X_train))
print("Y_train max:", np.max(Y_train))
print("Y_train min:", np.min(Y_train))

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

import joblib
joblib.dump(sc, 'scaler.pkl')

## Building the ANN
# Initializing the ANN
ann = tf.keras.models.Sequential()

# Adding layers
ann.add(tf.keras.layers.Dense(units=64, activation='relu'))
ann.add(tf.keras.layers.Dense(units=64, activation='relu'))
ann.add(tf.keras.layers.Dense(units=64, activation='relu'))
ann.add(tf.keras.layers.Dense(units=64, activation='relu'))
ann.add(tf.keras.layers.Dense(units=1))

## Training the ANN
# Compiling the ANN
ann.compile(optimizer='adam', loss='mean_squared_error')

# Training the ANN on the Training set
ann.fit(X_train, Y_train, batch_size = 32, epochs = 100)

## Making the predictions and evaluating the model
# Predicting the Test set results
Y_pred = ann.predict(X_test)

## Evaluating the Model
from sklearn.metrics import mean_squared_error, r2_score

# Average Mean Squared Error
mse_avg = mean_squared_error(Y_test, Y_pred, multioutput='uniform_average')
print(f'Average Mean Squared Error across all outputs: {mse_avg}')

# R-squared Score
r2 = r2_score(Y_test, Y_pred, multioutput='uniform_average')
print(f'R-squared score: {r2}')

# Saving the model to a file
ann.save('my_ann_model.h5')