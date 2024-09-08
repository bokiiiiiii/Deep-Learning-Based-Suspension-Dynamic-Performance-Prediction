import numpy as np
import pandas as pd
import tensorflow as tf

import joblib
# joblib.dump(sc, 'scaler.pkl')
sc = joblib.load('scaler.pkl')

# Importing the dataset
dataset = pd.read_excel('Test.xlsx')
X_sim = dataset.iloc[:, :13].values
Y_sim = dataset.iloc[:, 29].values
X_plt = X_sim[:, 0]
X_sim = sc.transform(X_sim)

# Loading the model from the file
loaded_model = tf.keras.models.load_model('my_ann_model.h5')

# Making predictions with the loaded model
Y_pred = loaded_model.predict(X_sim)

import matplotlib.pyplot as plt
Y_pred = Y_pred.flatten()

plt.figure(figsize=(10, 6))


plt.plot(X_plt, Y_sim, color='blue', label='Simulation Values')
plt.plot(X_plt, Y_pred, color='red', label='Predicted Values')
plt.title('Simulation vs Predicted Values')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()